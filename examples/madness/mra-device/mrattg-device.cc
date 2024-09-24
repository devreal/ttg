#include <ttg.h>
#include "tensor.h"
#include "tensorview.h"
#include "functionnode.h"
#include "functiondata.h"
#include "kernels.h"
#include "gaussian.h"
#include "functionfunctor.h"
#include "key.h"
#include "domain.h"

#include <ttg/serialization/backends.h>
#include <ttg/serialization/std/array.h>

#ifdef TTG_ENABLE_HOST
#define TASKTYPE void
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::Host;
#else
#define TASKTYPE ttg::device::Task
constexpr const ttg::ExecutionSpace Space = ttg::ExecutionSpace::CUDA;
#endif


template <mra::Dimension NDIM>
auto make_start(const ttg::Edge<mra::Key<NDIM>, void>& ctl) {
    auto func = [](const mra::Key<NDIM>& key) { ttg::sendk<0>(key); };
    return ttg::make_tt<mra::Key<NDIM>>(func, ttg::edges(), edges(ctl), "start", {}, {"control"});
}

template<typename FnT, typename T, mra::Dimension NDIM>
auto make_project(
  mra::Domain<NDIM>& domain,
  ttg::Buffer<FnT>& fb,
  std::size_t K,
  const mra::FunctionData<T, NDIM>& functiondata,
  const T thresh, /// should be scalar value not complex
  ttg::Edge<mra::Key<NDIM>, void> control,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionReconstructedNode<T, NDIM>> result)
{
  /* create a non-owning buffer for domain and capture it */
  auto fn = [&, K, thresh, db = ttg::Buffer<mra::Domain<NDIM>>(&domain), gl = mra::GLbuffer<T>()]
            (const mra::Key<NDIM>& key) -> TASKTYPE {
    using tensor_type = typename mra::Tensor<T, NDIM>;
    using key_type = typename mra::Key<NDIM>;
    using node_type = typename mra::FunctionReconstructedNode<T, NDIM>;
    auto result = node_type(key, K);
    tensor_type& coeffs = result.coeffs;
    auto outputs = ttg::device::forward();
    auto& f = *fb.host_ptr();

    if (key.level() < initial_level(f)) {
      std::vector<mra::Key<NDIM>> bcast_keys;
      /* TODO: children() returns an iteratable object but broadcast() expects a contiguous memory range.
                We need to fix broadcast to support any ranges */
      for (auto child : children(key)) bcast_keys.push_back(child);

#ifndef TTG_ENABLE_HOST
      outputs.push_back(ttg::device::broadcastk<0>(std::move(bcast_keys)));
#else
      ttg::broadcastk<0>(std::move(bcast_keys));
#endif
      coeffs.current_view() = T(1e7); // set to obviously bad value to detect incorrect use
      result.is_leaf = false;
    }
    else if (mra::is_negligible<FnT,T,NDIM>(f, domain.template bounding_box<T>(key), mra::truncate_tol(key,thresh))) {
      /* zero coeffs */
      coeffs.current_view() = T(0.0);
      result.is_leaf = true;
    }
    else {
      /* here we actually compute: first select a device */
      //result.is_leaf = fcoeffs(f, functiondata, key, thresh, coeffs);
      /**
       * BEGIN FCOEFFS HERE
       * TODO: figure out a way to outline this into a function or coroutine
       */

      /* global function data */
      // TODO: need to make our own FunctionData with dynamic K
      const auto& phibar = functiondata.get_phibar();
      const auto& hgT = functiondata.get_hgT();

      /* temporaries */
      bool is_leaf;
      auto is_leaf_scratch = ttg::make_scratch(&is_leaf, ttg::scope::Allocate);
      const std::size_t tmp_size = project_tmp_size<NDIM>(K);
      auto tmp = std::make_unique_for_overwrite<T[]>(tmp_size); // TODO: move this into make_scratch()
      auto tmp_scratch = ttg::make_scratch(tmp.get(), ttg::scope::Allocate, tmp_size);

      /* TODO: cannot do this from a function, need to move it into the main task */
#ifndef TTG_ENABLE_HOST
      co_await ttg::device::select(db, gl, fb, coeffs.buffer(), phibar.buffer(),
                                   hgT.buffer(), tmp_scratch, is_leaf_scratch);
#endif
      auto coeffs_view = coeffs.current_view();
      auto phibar_view = phibar.current_view();
      auto hgT_view    = hgT.current_view();
      T* tmp_device = tmp_scratch.device_ptr();
      bool *is_leaf_device = is_leaf_scratch.device_ptr();
      FnT* f_ptr   = fb.current_device_ptr();
      auto& domain = *db.current_device_ptr();
      auto  gldata = gl.current_device_ptr();

      /* submit the kernel */
      submit_fcoeffs_kernel(domain, gldata, *f_ptr, key, coeffs_view,
                            phibar_view, hgT_view, tmp_device,
                            is_leaf_device, thresh, ttg::device::current_stream());

      /* wait and get is_leaf back */
#ifndef TTG_ENABLE_HOST
      co_await ttg::device::wait(is_leaf_scratch);
#endif
      result.is_leaf = is_leaf;

      /**
       * END FCOEFFS HERE
       */

      if (!result.is_leaf) {
        std::vector<mra::Key<NDIM>> bcast_keys;
        for (auto child : children(key)) bcast_keys.push_back(child);
#ifndef TTG_ENABLE_HOST
        outputs.push_back(ttg::device::broadcastk<0>(std::move(bcast_keys)));
#else
        ttg::broadcastk<0>(bcast_keys);
#endif
      }
    }
#ifndef TTG_ENABLE_HOST
    outputs.push_back(ttg::device::send<1>(key, std::move(result))); // always produce a result
    co_await std::move(outputs);
#else
    ttg::send<1>(key, std::move(result));
#endif
  };

  ttg::Edge<mra::Key<NDIM>, void> refine("refine");
  return ttg::make_tt<Space>(std::move(fn), ttg::edges(fuse(control, refine)), ttg::edges(refine,result), "project");
}

template<mra::Dimension NDIM, typename Value, std::size_t I, std::size_t... Is>
static auto select_compress_send(const mra::Key<NDIM>& key, Value&& value,
                                 std::size_t child_idx,
                                 std::index_sequence<I, Is...>) {
  if (child_idx == I) {
#ifndef TTG_ENABLE_HOST
    return ttg::device::send<I>(key.parent(), std::forward<Value>(value));
#else
    return ttg::send<I>(key.parent(), std::forward<Value>(value));
#endif
  } else if constexpr (sizeof...(Is) > 0){
    return select_compress_send(key, std::forward<Value>(value), child_idx, std::index_sequence<Is...>{});
  }
  /* if we get here we messed up */
  throw std::runtime_error("Mismatching number of children!");
}

/* forward a reconstructed function node to the right input of do_compress
 * this is a device task to prevent data from being pulled back to the host
 * even though it will not actually perform any computation */
template<typename T, mra::Dimension NDIM>
static TASKTYPE do_send_leafs_up(const mra::Key<NDIM>& key, const mra::FunctionReconstructedNode<T, NDIM>& node) {
  /* drop all inputs from nodes that are not leafs, they will be upstreamed by compress */
  if (!node.has_children()) {
#ifndef TTG_ENABLE_HOST
    co_await select_compress_send(key, node, key.childindex(), std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
#else
    select_compress_send(key, node, key.childindex(), std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
#endif
  }
}


/// Make a composite operator that implements compression for a single function
template <typename T, mra::Dimension NDIM>
static auto make_compress(
  const std::size_t K,
  const mra::FunctionData<T, NDIM>& functiondata,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionReconstructedNode<T, NDIM>>& in,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionCompressedNode<T, NDIM>>& out)
{
  static_assert(NDIM == 3); // TODO: worth fixing?

  constexpr const std::size_t num_children = mra::Key<NDIM>::num_children();
  // creates the right number of edges for nodes to flow from send_leafs_up to compress
  // send_leafs_up will select the right input for compress
  auto create_edges = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return ttg::edges(((void)Is, ttg::Edge<mra::Key<NDIM>, mra::FunctionReconstructedNode<T, NDIM>>{})...);
  };
  auto send_to_compress_edges = create_edges(std::make_index_sequence<num_children>{});
  /* append out edge to set of edges */
  auto compress_out_edges = std::tuple_cat(send_to_compress_edges, std::make_tuple(out));
  /* use the tuple variant to handle variable number of inputs while suppressing the output tuple */
  auto do_compress = [&, K](const mra::Key<NDIM>& key,
                         //const std::tuple<const FunctionReconstructedNodeTypes&...>& input_frns
                         const mra::FunctionReconstructedNode<T,NDIM> &in0,
                         const mra::FunctionReconstructedNode<T,NDIM> &in1,
                         const mra::FunctionReconstructedNode<T,NDIM> &in2,
                         const mra::FunctionReconstructedNode<T,NDIM> &in3,
                         const mra::FunctionReconstructedNode<T,NDIM> &in4,
                         const mra::FunctionReconstructedNode<T,NDIM> &in5,
                         const mra::FunctionReconstructedNode<T,NDIM> &in6,
                         const mra::FunctionReconstructedNode<T,NDIM> &in7) -> TASKTYPE {
    //const typename ::detail::tree_types<T,K,NDIM>::compress_in_type& in,
    //typename ::detail::tree_types<T,K,NDIM>::compress_out_type& out) {
      constexpr const auto num_children = mra::Key<NDIM>::num_children();
      constexpr const auto out_terminal_id = num_children;
      mra::FunctionCompressedNode<T,NDIM> result(key, K); // The eventual result
      auto& d = result.coeffs;
      // allocate even though we might not need it
      mra::FunctionReconstructedNode<T, NDIM> p(key, K);

      /* stores sumsq for each child and for result at the end of the kernel */
      const std::size_t tmp_size = compress_tmp_size<NDIM>(K);
      auto tmp = std::make_unique_for_overwrite<T[]>(tmp_size);
      const auto& hgT = functiondata.get_hgT();
      auto tmp_scratch = ttg::make_scratch(tmp.get(), ttg::scope::Allocate, tmp_size);
      T sumsqs[num_children+1];
      auto sumsqs_scratch = ttg::make_scratch(sumsqs, ttg::scope::Allocate, num_children+1);
#ifndef TTG_ENABLE_HOST
      co_await ttg::device::select(p.coeffs.buffer(), d.buffer(), hgT.buffer(),
                                   tmp_scratch, sumsqs_scratch,
                                   in0.coeffs.buffer(), in1.coeffs.buffer(),
                                   in2.coeffs.buffer(), in3.coeffs.buffer(),
                                   in4.coeffs.buffer(), in5.coeffs.buffer(),
                                   in6.coeffs.buffer(), in7.coeffs.buffer());
#endif

      /* assemble input array and submit kernel */
      //auto input_ptrs = std::apply([](auto... ins){ return std::array{(ins.coeffs.buffer().current_device_ptr())...}; });
      auto input_ptrs = std::array{in0.coeffs.buffer().current_device_ptr(), in1.coeffs.buffer().current_device_ptr(),
                                   in2.coeffs.buffer().current_device_ptr(), in3.coeffs.buffer().current_device_ptr(),
                                   in4.coeffs.buffer().current_device_ptr(), in5.coeffs.buffer().current_device_ptr(),
                                   in6.coeffs.buffer().current_device_ptr(), in7.coeffs.buffer().current_device_ptr()};

      auto coeffs_view = p.coeffs.current_view();
      auto rcoeffs_view = d.current_view();
      auto hgT_view = hgT.current_view();

      submit_compress_kernel(key, coeffs_view, rcoeffs_view, hgT_view,
                            tmp_scratch.device_ptr(), sumsqs_scratch.device_ptr(), input_ptrs,
                            ttg::device::current_stream());

      /* wait for kernel and transfer sums back */
#ifndef TTG_ENABLE_HOST
      co_await ttg::device::wait(sumsqs_scratch);
#endif
      T sumsq = 0.0;
      T d_sumsq = tmp[num_children];
      {  // Collect child leaf info
        //result.is_child_leaf = std::apply([](auto... ins){ return std::array{(ins.is_leaf)...}; });
        result.is_child_leaf = std::array{in0.is_leaf, in1.is_leaf, in2.is_leaf, in3.is_leaf,
                                          in4.is_leaf, in5.is_leaf, in6.is_leaf, in7.is_leaf};
        auto sumsqs = std::array{in0.sum, in1.sum, in2.sum, in3.sum,
                                 in4.sum, in5.sum, in6.sum, in7.sum};
        sumsq = std::reduce(sumsqs.begin(), sumsqs.end());
      }

      // Recur up
      std::cout << "compress key " << key << " parent " << key.parent() << " level " << key.level()
                << " sumsq " << sumsq << " d_sumsq " << d_sumsq << std::endl;
      if (key.level() > 0) {
        p.sum = d_sumsq + sumsq; // result sumsq is last element in sumsqs

        // will not return
#ifndef TTG_ENABLE_HOST
        co_await ttg::device::forward(
          // select to which child of our parent we send
          //ttg::device::send<0>(key, std::move(p)),
          select_compress_send(key, std::move(p), key.childindex(), std::make_index_sequence<num_children>{}),
          // Send result to output tree
          ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
          select_compress_send(key, std::move(p), key.childindex(), std::make_index_sequence<num_children>{});
          ttg::send<out_terminal_id>(key, std::move(result));
#endif
      } else {
        std::cout << "At root of compressed tree: total normsq is " << sumsq + d_sumsq << std::endl;
#ifndef TTG_ENABLE_HOST
        co_await ttg::device::forward(
          // Send result to output tree
          ttg::device::send<out_terminal_id>(key, std::move(result)));
#else
        ttg::send<out_terminal_id>(key, std::move(result));
#endif
      }
  };
  ttg::Edge<mra::Key<NDIM>, mra::FunctionReconstructedNode<T, NDIM>> recur("recur");
  return std::make_tuple(ttg::make_tt<Space>(&do_send_leafs_up<T,NDIM>, edges(in), send_to_compress_edges, "send_leaves_up"),
                         ttg::make_tt<Space>(std::move(do_compress), send_to_compress_edges, compress_out_edges, "do_compress"));
}

template <typename T, mra::Dimension NDIM>
auto make_reconstruct(
  const std::size_t K,
  const mra::FunctionData<T, NDIM>& functiondata,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionCompressedNode<T, NDIM>> in,
  ttg::Edge<mra::Key<NDIM>, mra::FunctionReconstructedNode<T, NDIM>> out,
  const std::string& name = "reconstruct")
{
  ttg::Edge<mra::Key<NDIM>, mra::Tensor<T,NDIM>> S("S");  // passes scaling functions down

  auto do_reconstruct = [&, K](const mra::Key<NDIM>& key,
                            mra::FunctionCompressedNode<T, NDIM>&& node,
                            const mra::Tensor<T, NDIM>& from_parent) -> TASKTYPE {
    const std::size_t K = from_parent.dim(0);
    const std::size_t tmp_size = reconstruct_tmp_size<NDIM>(K);
    auto tmp = std::make_unique_for_overwrite<T[]>(tmp_size);
    const auto& hg = functiondata.get_hg();
    auto tmp_scratch = ttg::make_scratch(tmp.get(), ttg::scope::Allocate, tmp_size);

    // Send empty interior node to result tree
    auto r_empty = mra::FunctionReconstructedNode<T,NDIM>(key, K);
    r_empty.coeffs.current_view() = T(0.0);
    r_empty.is_leaf = false;

    /* populate the vector of r's */
    std::array<mra::FunctionReconstructedNode<T,NDIM>, mra::Key<NDIM>::num_children()> r_arr;
    for (int i = 0; i < key.num_children(); ++i) {
      r_arr[i] = mra::FunctionReconstructedNode<T,NDIM>(key, K);
    }

    // helper lambda to pick apart the std::array
    auto do_select = [&]<std::size_t... Is>(std::index_sequence<Is...>){
      return ttg::device::select(hg.buffer(), from_parent.buffer(),
                                 node.coeffs.buffer(), tmp_scratch,
                                 (r_arr[Is].coeffs.buffer())...);
    };
    /* select a device */
#ifndef TTG_ENABLE_HOST
    co_await do_select(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
#endif

    // helper lambda to pick apart the std::array
    auto assemble_tensor_ptrs = [&]<std::size_t... Is>(std::index_sequence<Is...>){
      return std::array{(r_arr[Is].coeffs.current_view().data())...};
    };
    auto r_ptrs = assemble_tensor_ptrs(std::make_index_sequence<mra::Key<NDIM>::num_children()>{});
    auto node_view = node.coeffs.current_view();
    auto hg_view = hg.current_view();
    auto from_parent_view = from_parent.current_view();
    submit_reconstruct_kernel(key, node_view, hg_view, from_parent_view,
                              r_ptrs, tmp_scratch.device_ptr(), K, ttg::device::current_stream());

    // forward() returns a vector that we can push into
#ifndef TTG_ENABLE_HOST
    auto sends = ttg::device::forward(ttg::device::send<1>(key, std::move(r_empty)));
    mra::KeyChildren<NDIM> children(key);
    for (auto it=children.begin(); it!=children.end(); ++it) {
        const mra::Key<NDIM> child= *it;
        mra::FunctionReconstructedNode<T,NDIM>& r = r_arr[it.index()];
        r.key = child;
        r.is_leaf = node.is_child_leaf[it.index()];
        if (r.is_leaf) {
          sends.push_back(ttg::device::send<1>(child, std::move(r)));
        }
        else {
          sends.push_back(ttg::device::send<0>(child, std::move(r.coeffs)));
        }
    }
    co_await std::move(sends);
#else
    ttg::send<1>(key, std::move(r_empty));
    mra::KeyChildren<NDIM> children(key);
    for (auto it=children.begin(); it!=children.end(); ++it) {
        const mra::Key<NDIM> child= *it;
        mra::FunctionReconstructedNode<T,NDIM>& r = r_arr[it.index()];
        r.key = child;
        r.is_leaf = node.is_child_leaf[it.index()];
        if (r.is_leaf) {
          ttg::send<1>(child, std::move(r));
        }
        else {
          ttg::send<0>(child, std::move(r.coeffs));
        }
    }
#endif // TTG_ENABLE_HOST
  };


  auto s = ttg::make_tt<Space>(std::move(do_reconstruct), ttg::edges(in, S), ttg::edges(S, out), name, {"input", "s"}, {"s", "output"});

  if (ttg::default_execution_context().rank() == 0) {
    s->template in<1>()->send(mra::Key<NDIM>{0,{0}}, mra::Tensor<T,NDIM>(K)); // Prime the flow of scaling functions
  }

  return s;
}


static std::mutex printer_guard;
template <typename keyT, typename valueT>
auto make_printer(const ttg::Edge<keyT, valueT>& in, const char* str = "", const bool doprint=true) {
  auto func = [str,doprint](const keyT& key, const valueT& value) {
    if (doprint) {
      std::lock_guard<std::mutex> obolus(printer_guard);
      std::cout << str << " (" << key << "," << value << ")" << std::endl;
    }
  };
  return ttg::make_tt(func, ttg::edges(in), ttg::edges(), "printer", {"input"});
}

template<typename T, mra::Dimension NDIM>
void test(std::size_t K) {
  auto functiondata = mra::FunctionData<T,NDIM>(K);
  mra::Domain<NDIM> D;
  D.set_cube(-6.0,6.0);

  srand48(5551212); // for reproducible results
  for (int i = 0; i < 10000; ++i) drand48(); // warmup generator

  ttg::Edge<mra::Key<NDIM>, void> project_control;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionReconstructedNode<T, NDIM>> project_result, reconstruct_result;
  ttg::Edge<mra::Key<NDIM>, mra::FunctionCompressedNode<T, NDIM>> compress_result;

  // define a Gaussian
  //auto gaussian = mra::Gaussian<T, NDIM>(D, T(3.0), {T(0.0),T(0.0),T(0.0)});
  T expnt = 30000.0;
  mra::Coordinate<T,NDIM> r;
  for (size_t d=0; d<NDIM; d++) {
    r[d] = T(-6.0) + T(12.0)*drand48();
  }
  auto gaussian = mra::Gaussian<T,NDIM>(D, expnt, r);

  // put it into a buffer
  auto gauss_buffer = ttg::Buffer<mra::Gaussian<T, NDIM>>(&gaussian);
  auto start = make_start(project_control);
  auto project = make_project(D, gauss_buffer, K, functiondata, T(1e-6), project_control, project_result);
  auto compress = make_compress(K, functiondata, project_result, compress_result);
  auto reconstruct = make_reconstruct(K, functiondata, compress_result, reconstruct_result);
  auto printer =   make_printer(project_result,    "projected    ", false);
  auto printer2 =  make_printer(compress_result,   "compressed   ", false);
  auto printer3 =  make_printer(reconstruct_result,"reconstructed", false);

  auto connected = make_graph_executable(start.get());
  assert(connected);

  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  if (ttg::default_execution_context().rank() == 0) {
      //std::cout << "Is everything connected? " << connected << std::endl;
      //std::cout << "==== begin dot ====\n";
      //std::cout << Dot()(start.get()) << std::endl;
      //std::cout << "====  end dot  ====\n";

      beg = std::chrono::high_resolution_clock::now();
      // This kicks off the entire computation
      start->invoke(mra::Key<NDIM>(0, {0}));
  }
  ttg::execute();
  ttg::fence();

  if (ttg::default_execution_context().rank() == 0) {
    end = std::chrono::high_resolution_clock::now();
    std::cout << "TTG Execution Time (milliseconds) : "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count()) / 1000
              << std::endl;
  }
}

int main(int argc, char **argv) {
  ttg::initialize(argc, argv);
  mra::GLinitialize();

  test<double, 3>(3);

  ttg::finalize();
}