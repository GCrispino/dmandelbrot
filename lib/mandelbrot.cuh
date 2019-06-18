#ifndef MANDELBROT_COMMON_H
#   include "mandelbrot_common.cuh"
#endif

#ifdef __CUDACC__
#   define __DEVICE__ __device__
#   define __HOST__ __host__
#   define MANDELBROT_FN mandelbrot_gpu
#   include "mandelbrot_gpu.cuh"
#else
#   define __DEVICE__
#   define __HOST__
#   define MANDELBROT_FN mandelbrot_cpu
#   define COMPLEX std 
#endif

#include <iostream>
#include <omp.h>

namespace mandelbrot{
    using COMPLEX::complex;

    enum exec_mode{
        CPU = 0,
        GPU = 1
    };

    void mandelbrot_cpu(
        unsigned n_threads,
        complex<REAL_TYPE> c0, complex<REAL_TYPE> c1,
        REAL_TYPE delta_x, REAL_TYPE delta_y,
        unsigned w, unsigned h, unsigned m,
        unsigned *table
    ){

        omp_set_num_threads(n_threads);

        #pragma omp parallel for
        for (unsigned i = 0; i < w * h; ++i){
            unsigned pixel_y = i / w;
            REAL_TYPE y = c0.imag() + pixel_y * delta_y;

            unsigned pixel_x = i % w;
            REAL_TYPE x = c0.real() + pixel_x * delta_x;
            table[i] = mandelbrot_c(complex<REAL_TYPE>(x,y),m);
        }

    }

    void mandelbrot(
        exec_mode ex, unsigned n_threads,
        complex<REAL_TYPE> c0, complex<REAL_TYPE> c1,
        REAL_TYPE delta_x, REAL_TYPE delta_y,
        unsigned w, unsigned h, unsigned m,
        unsigned *table
    ){

        std::cout << "Delta x: " << delta_x << std::endl;
        std::cout << "Delta y: " << delta_y << std::endl;

        std::cout << "c0: (" << c0.real() << ',' << c0.imag() << ")" << std::endl;
        std::cout << "c1: (" << c1.real() << ',' << c1.imag() << ")" << std::endl;
        std::cout << "Exec mode: " << ex << std::endl;

        switch(ex){
            case exec_mode::CPU:
                mandelbrot_cpu(
                    n_threads,
                    c0, c1,
                    delta_x, delta_y,
                    w, h, m,
                    table
                );
                break;
            case exec_mode::GPU:
                MANDELBROT_FN(
                    n_threads,
                    c0, c1,
                    delta_x, delta_y,
                    w, h, m,
                    table
                );
                break;
        }

    }

    int * get_boundaries(int w, int h, int rank, int n_procs, int block_size, bool print = false){
        int 
            // n_blocks_per_line = w / block_size,
            n_blocks_per_column = h / block_size;
        int
            block_line = (rank - 1) % n_blocks_per_column,
            // block_column = rank % n_blocks_per_line,
            // start_x = block_size * block_column,
            start_x = 0,
            end_x = w - 1,
            // end_x = start_x + block_size < w - 1 ? start_x + block_size : w - 1,
            start_y = block_size * block_line,
            // start_y = 0,
            // end_y = h - 1,
            end_y = rank == n_procs ? h - 1 : start_y + block_size - 1,
            // end_y = start_y + block_size - 1 < h - 1 ? start_y + block_size - 1 : h - 1,
            *res = (int *) malloc(4 * sizeof(int));

        if (print){
            printf("\tn_procs: %d\n",n_procs);
            printf("\trank %d, x: (%d, %d), y: (%d, %d)\n", rank, start_x, end_x, start_y, end_y);
        }
        res[0] = start_x;
        res[1] = end_x;
        res[2] = start_y;
        res[3] = end_y;

        return res;
    }

    unsigned get_block_size(unsigned w, unsigned h, unsigned n_procs){
        unsigned block_size = 1;
        while (block_size * 2 < (w / n_procs) && block_size * 2 < (h / n_procs)){
            block_size *= 2;
        }

        return block_size;
    }

    void d_mandelbrot(
        exec_mode ex, unsigned n_threads,
        unsigned rank, unsigned n_procs,
        complex<REAL_TYPE> c0, complex<REAL_TYPE> c1,
        REAL_TYPE delta_x, REAL_TYPE delta_y,
        unsigned w, unsigned h, unsigned m,
        unsigned *table
    ){

        unsigned block_size = get_block_size(w, h, n_procs);
        // printf("block_size %d\n",block_size);
        int *bound = get_boundaries(w, h, rank, n_procs, block_size, true);
        int start_y = bound[2], end_y = bound[3];
        unsigned local_h = end_y - start_y;
        unsigned local_table_size = w * local_h;
        unsigned *local_table = new unsigned[local_table_size];
        complex<REAL_TYPE> local_c0(c0.real(), c0.imag() + start_y * delta_y), local_c1(c1.real(), c1.imag() + start_y * delta_y);

        mandelbrot::mandelbrot(
            ex, n_threads,
            local_c0, local_c1,
            delta_x, delta_y,
            w, local_h, m, local_table
        );

        printf("\tRank: %d will send %d!\n", rank, local_table_size);
        MPI_Send(
            local_table,
            local_table_size,
            MPI_UNSIGNED,
            0,
            0,
            MPI_COMM_WORLD
        );
        printf("\tRank: %d sent data!\n", rank);
    }
}
