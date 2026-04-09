#include "getrf.h"




int main(int argc, char **argv) {


  std::chrono::time_point<std::chrono::high_resolution_clock> beg, end;
  int NB = 32;
  int N = 5*NB;
  int M = N;
  int nthreads = -1;
  const char* prof_filename = nullptr;
  char *opt = nullptr;
  int ret = EXIT_SUCCESS;
  int niter = 3;
  bool print_dot = false;

  if( (opt = getCmdOption(argv+1, argv+argc, "-N")) != nullptr ) {
    N = M = atoi(opt);
  }

  if( (opt = getCmdOption(argv+1, argv+argc, "-t")) != nullptr ) {
    NB = atoi(opt);
  }

  if( (opt = getCmdOption(argv+1, argv+argc, "-c")) != nullptr ) {
    nthreads = atoi(opt);
  }

  if( (opt = getCmdOption(argv+1, argv+argc, "-dag")) != nullptr ) {
    prof_filename = opt;
  }

  if( (opt = getCmdOption(argv+1, argv+argc, "-n")) != nullptr) {
    niter = atoi(opt);
  }

  /* whether to print the TTG dot */
  print_dot = cmdOptionExists(argv+1, argv+argc, "-dot");


  bool check = !cmdOptionExists(argv+1, argv+argc, "-x");

  /* whether we set a device mapping */
  bool enable_device_map = !cmdOptionExists(argv, argv+argc, "--default-device-map");

  // TODO: need to filter out our arguments to make parsec happy
  ttg::initialize(1, argv, nthreads);

  /* set up TA to get the allocator */
  allocator_init(argc, argv);

  auto world = ttg::default_execution_context();
  if(nullptr != prof_filename) {
    world.profile_on();
    world.dag_on(prof_filename);
  }

  int P = std::sqrt(world.size());
  int Q = (world.size() + P - 1)/P;

  if ( (opt = getCmdOption(argv+1, argv + argc, "-P")) != nullptr) {
    P = atoi(opt);
  }
  if ( (opt = getCmdOption(argv+1, argv + argc, "-Q")) != nullptr) {
    Q = atoi(opt);
  }


  if(check && (P>1 || Q>1)) {
    std::cerr << "Check is disabled for distributed runs at this time" << std::endl;
    check = false;
  }

  if (world.rank() == 0) {
    std::cout << "Creating 2D block cyclic matrix with NB " << NB << " N " << N << " M " << M << " P " << P << " Q " << Q << std::endl;
  }

  parsec_matrix_block_cyclic_t dcA;
  parsec_matrix_block_cyclic_init(&dcA, parsec_matrix_type_t::PARSEC_MATRIX_DOUBLE,
                                  world.rank(), NB, NB, N, M,
                                  0, 0, N, M, P, Q, PARSEC_MATRIX_LOWER);
  dcA.mat = parsec_data_allocate((size_t)dcA.super.nb_local_tiles *
                                 (size_t)dcA.super.bsiz *
                                 (size_t)parsec_datadist_getsizeoftype(dcA.super.mtype));

  /* would be nice to have proper abstractions for this */
  parsec_data_collection_t *o = &(dcA.super.super);
  for (int devid = 1; devid < parsec_nb_devices; ++devid) {
    auto* device = parsec_mca_device_get(devid);
    if (device->memory_register) {
      o->register_memory(o, device); // TODO: check device IDs
    }
  }

  parsec_data_collection_set_key((parsec_data_collection_t*)&dcA, (char*)"Matrix A");


  for (int i = 0; i < niter; ++i) {
    parsec_devices_release_memory();

    //Matrix<double>* A = new Matrix<double>(n_rows, n_cols, NB, NB);
    MatrixT<double> A{&dcA};
    /* TODO: initialize the matrix */
    /* This works only with the parsec backend! */
    int random_seed = 3872;

    init_matrix(A, random_seed);
    ttg::Edge<Key2, MatrixTile<double>> startup("startup");
    ttg::Edge<Key2, MatrixTile<double>> result("To result");

    auto getrf_init_tt = make_load_tt(A, startup);
    auto getrf_ttg = make_getrf_ttg(A, startup, result, enable_device_map);
    auto getrf_result_ttg = make_result_ttg(A, result);

    auto connected = make_graph_executable(getrf_init_tt.get());
    assert(connected);
    TTGUNUSED(connected);

    if (world.rank() == 0) {
      if (print_dot) {
        std::cout << "==== begin dot ====\n";
        std::cout << ttg::Dot()(getrf_init_tt.get()) << std::endl;
        std::cout << "==== end dot ====\n";
      }
      beg = std::chrono::high_resolution_clock::now();
    }

    if (world.rank() == 0) {
      beg = std::chrono::high_resolution_clock::now();
    }

    getrf_init_tt->invoke();
    ttg::execute(world);
    ttg::fence(world);

    if (world.rank() == 0) {
      end = std::chrono::high_resolution_clock::now();
      auto elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
      end = std::chrono::high_resolution_clock::now();
      std::cout << "TTG Execution Time (milliseconds) : "
                << elapsed / 1E3 << " : Flops " << (potrf::FLOPS_DPOTRF(N)) << " " << (potrf::FLOPS_DPOTRF(N)/1e9)/(elapsed/1e6) << " GF/s" << std::endl;
    }
#if defined(TTG_PARSEC_IMPORTED)
    parsec_devices_reset_load(ttg::default_execution_context().impl().context());
#endif // TTG_PARSEC_IMPORTED
  }

  world.dag_off();

  ttg::finalize();

}