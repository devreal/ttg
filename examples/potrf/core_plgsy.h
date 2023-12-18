#pragma once

#include <complex>

/**
 *
 * @file core_dplgsy.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.8.0
 * @author Piotr Luszczek
 * @author Pierre Lemarinier
 * @author Mathieu Faverge
 * @date 2010-11-15
 * @generated d Tue Feb 23 09:48:16 2021
 *
 **/
#include <stdint.h>
#include "random.h"

/***************************************************************************//**
 *
 * @ingroup dplasma_cores_double
 *
 *  CORE_dplgsy generates a symmetric matrix.
 *
 *******************************************************************************
 *
 * @param[in] bump
 *         Scalar added to the diagonal of the full Matrix A to make it diagonal
 *         dominant.
 *
 * @param[in] m
 *         The number of rows of the tile A. m >= 0.
 *
 * @param[in] n
 *         The number of columns of the tile A. n >= 0.
 *
 * @param[in,out] A
 *         On entry, the m-by-n tile to be initialized.
 *         On exit, the tile initialized in the mtxtype format.
 *
 * @param[in] lda
 *         The leading dimension of the tile A. lda >= max(1,m).
 *
 * @param[in] gM
 *         The global number of rows of the full matrix, A is belonging to. gM >= (m0+M).
 *
 * @param[in] m0
 *         The index of the first row of tile A in the full matrix. m0 >= 0.
 *
 * @param[in] n0
 *         The index of the first column of tile A in the full matrix. n0 >= 0.
 *
 * @param[in] seed
 *         The seed used for random generation. Must be the same for
 *         all tiles initialized with this routine.
 *
 ******************************************************************************/
template <typename T>
void CORE_plgsy( T bump, int m, int n, T *A, int lda,
                 int gM, int m0, int n0, unsigned long long int seed )
{
    T *tmp = A;
    int64_t i, j;
    unsigned long long int ran, jump;
    int nbelem = 1;
    if constexpr (std::is_same<T, std::complex<double>>::value ||
        std::is_same<T, std::complex<float>>::value ) {
        nbelem = 2;
    }

    jump = (unsigned long long int)m0 + (unsigned long long int)n0 * (unsigned long long int)gM;


    auto fill_line = [](T* line, int from, int to, int ran, int ptr_advance){
            T* tmp = line;
            for (int i = from; i < to; i++) {
                *tmp = 0.5f - ran * RndF_Mul;
                ran  = Rnd64_A * ran + Rnd64_C;
                if constexpr (std::is_same<T, std::complex<double>>::value ||
                    std::is_same<T, std::complex<float>>::value ) {
                    *tmp += T(- ran * RndF_Mul, 0.5);
                    ran   = Rnd64_A * ran + Rnd64_C;
                }
                tmp += ptr_advance;
            }
        };
    /*
     * Tile diagonal
     */
    if ( m0 == n0 ) {
        for (j = 0; j < n; j++) {
            ran = Rnd64_jump( nbelem * jump, seed );
            fill_line(tmp, j, m, ran, 1);
            tmp  += (lda - i + j + 1);
            jump += gM + 1;
        }

        for (j = 0; j < n; j++) {
            A[j+j*lda] += bump;

            for (i=0; i<j; i++) {
                A[lda*j+i] = A[lda*i+j];
            }
        }
    }
    /*
     * Lower part
     */
    else if ( m0 > n0 ) {
        for (j = 0; j < n; j++) {
            ran = Rnd64_jump( nbelem * jump, seed );
            fill_line(tmp, 0, j, ran, 1);
            tmp  += (lda - i);
            jump += gM;
        }
    }
    /*
     * Upper part
     */
    else if ( m0 < n0 ) {
        /* Overwrite jump */
        jump = (unsigned long long int)n0 + (unsigned long long int)m0 * (unsigned long long int)gM;

        for (i = 0; i < m; i++) {
            ran = Rnd64_jump( nbelem * jump, seed );
            fill_line(&A[i], 0, n, ran, lda);
            jump += gM;
        }
    }
}


template <typename T>
void CORE_plgsy_device( T bump, int m, int n, T *A, int lda,
                        int gM, int m0, int n0, unsigned long long int seed );


template <typename T>
void CORE_plgsy_kokkos_host( T bump, int m, int n, T *A, int lda,
                             int gM, int m0, int n0, unsigned long long int seed );