#include <iostream>
#include <string>
#include <png++/image.hpp>
#include <mpi.h>

#define QUOTEME(x) QUOTEME_1(x)
#define QUOTEME_1(x) #x
#ifdef __CUDACC__
#   define  __CUDA__ 1
#   define INCLUDE_FILE(x) QUOTEME(thrust/complex.h)
#   define COMPLEX thrust
#else
#   define  __CUDA__ 0
#   define INCLUDE_FILE(x) QUOTEME(complex)
#   define COMPLEX std
#endif

#ifndef REAL_TYPE
#   define REAL_TYPE float
#endif

#include INCLUDE_FILE()

#include "mandelbrot.cuh"


void print_table(unsigned w, unsigned h, unsigned ** table){
    for (unsigned i = 0;i < h; ++i){
        for (unsigned j = 0;j < w; ++j){
            std::cout << table[i * h + j]  << ' ';
        }

        std::cout << std::endl;
    }
}

png::image<png::rgb_pixel> create_image(unsigned w, unsigned h, unsigned *table){

    png::image< png::rgb_pixel > image(w, h);

    #pragma omp parallel for
    for (png::uint_32 y = 0; y < image.get_height(); ++y)
    {
       for (png::uint_32 x = 0; x < image.get_width(); ++x)
       {
            if (table[y * w + x] == 0){
                image[y][x] = png::rgb_pixel(30, 30, 30);
            }
            else{
                image[y][x] = png::rgb_pixel(table[y * w + x] * 2, table[y * w + x] * 2, 170 + table[y * w + x] * 2);
            }
        }
    }

    return image;
}


struct params {
    COMPLEX::complex<REAL_TYPE> c0,c1;
    unsigned w,h,n_threads;
    mandelbrot::exec_mode ex;
    std::string output_path;

    params(
        const COMPLEX::complex<REAL_TYPE> &c0, const COMPLEX::complex<REAL_TYPE> &c1,
        unsigned w, unsigned h, unsigned n_threads,
        mandelbrot::exec_mode ex, const std::string &output_path
    ): c0(c0), c1(c1), w(w), h(h), n_threads(n_threads), ex(ex), output_path(output_path)
    {}
};

mandelbrot::exec_mode get_exec_mode(const char * mode){
    mandelbrot::exec_mode ex;
    if (mode[0] == '0'){
        ex = mandelbrot::exec_mode::CPU;
    }
    else if (mode[0] == '1'){
        if (!__CUDA__){
            std::cerr << "WARNING! You chose to use GPU execution without using nvcc" << std::endl;
            std::cerr << "\tDefaulting to CPU execution..." << std::endl;
            return mandelbrot::exec_mode::CPU;
        }
        ex = mandelbrot::exec_mode::GPU;
    }
    else{
        std::cerr << "Invalid execution mode (0 or 1 is allowed)!" << std::endl;
        exit(1);
    }

    return ex;
}

struct params parse_args(int argc, char **argv){

    using COMPLEX::complex;


    std::string
        usage("USAGE: dmbrot <C0_REAL_TYPE> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <OUTPUT>");

    if (argc != 10){
        std::cerr << usage << std::endl;
        exit(1);
    }

    REAL_TYPE
        c0_real = atof(argv[1]), c0_imag = atof(argv[2]),
        c1_real = atof(argv[3]), c1_imag = atof(argv[4]);

    unsigned w = atoi(argv[5]), h = atoi(argv[6]), n_threads = atoi(argv[8]);

    const complex<REAL_TYPE> c0(c0_real, c0_imag), c1(c1_real, c1_imag);

    return params(
        c0,c1,
        w, h, n_threads, get_exec_mode(argv[7]),argv[9]
    );
}


int main(int argc, char **argv){
    using COMPLEX::complex;

    MPI_Init(&argc, &argv);

    int world_size, world_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    params args = parse_args(argc, argv);

    const mandelbrot::exec_mode ex = args.ex;
    const unsigned w = args.w, h = args.h, m = 250;

    complex<REAL_TYPE> c0(args.c0),c1(args.c1);

    const REAL_TYPE delta_x = (c1.real() - c0.real()) / w;
    const REAL_TYPE delta_y = (c1.imag() - c0.imag()) / h;

    if(world_size > h + 1){
        if (world_rank == 0){
            std::cerr << "Number of processes cannot be higher than image height + 1!" << std::endl;
            std::cerr << "Setting number of processes to h + 1 = " << h + 1 << std::endl;
        }

        world_size = h + 1; 
    }

    world_size = world_size > h + 1 ? h + 1 : world_size;

    if (world_size == 1){
        if (world_rank == 0){
            std::cerr << "Number of processes needs to be at least 2! Exiting..." << std::endl;
        }

        MPI_Finalize();
        exit(1);
    }

    if (world_rank > world_size - 1){
        MPI_Finalize();
        exit(0);
    }

    if (world_rank == 0){
        std::cout << "Exec mode: " << ex << std::endl;

        std::cout << "c0: (" << c0.real() << ',' << c0.imag() << ")" << std::endl;
        std::cout << "c1: (" << c1.real() << ',' << c1.imag() << ")" << std::endl;
        std::cout << "w: " << w << ", h: " << h << std::endl;
        std::cout << "Delta x: " << delta_x << std::endl;
        std::cout << "Delta y: " << delta_y << std::endl;

        unsigned *table = new unsigned[w * h];
        unsigned block_size = mandelbrot::get_block_size(w, h, world_size - 1);

        for (unsigned i = 1; i < world_size; ++i){
            int *bound = mandelbrot::get_boundaries(w, h, i, world_size - 1, block_size);
            int start_x = bound[0], end_x = bound[1], start_y = bound[2], end_y = bound[3];
            unsigned local_h = end_y - start_y + 1;
            unsigned local_table_size = w * local_h;
            unsigned offset = block_size * w * (i - 1);

            MPI_Recv(
                table + offset,
                local_table_size,
                MPI_UNSIGNED,
                i,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        }
            
        png::image< png::rgb_pixel > image = create_image(w,h,table);
        image.write(args.output_path);

        delete[] table;
    }
    else{
        d_mandelbrot(
            ex, args.n_threads,
            world_rank, world_size - 1,
            c0, c1,
            delta_x, delta_y,
            w, h, m
        );
    }


    MPI_Finalize();

    return 0;
}
