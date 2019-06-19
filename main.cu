#include <iostream>
#include <string>
#include <functional>
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


template <class R, class ...A>
R do_master(int rank, std::function<R(A...)> fn, std::function<R(A...)> fn_else, A... args){
    if (rank == 0){ // master
        return fn(args...);
    }

    return fn_else(args...); // slave
}

void master_work(
    params args,
    int rank, int n_procs,
    REAL_TYPE delta_x, REAL_TYPE delta_y,
    unsigned m
){
    COMPLEX::complex<REAL_TYPE> c0(args.c0),c1(args.c1);
    const unsigned w = args.w, h = args.h;
    std::cout << "Exec mode: " << args.ex << std::endl;

    std::cout << "c0: (" << c0.real() << ',' << c0.imag() << ")" << std::endl;
    std::cout << "c1: (" << c1.real() << ',' << c1.imag() << ")" << std::endl;
    std::cout << "w: " << w << ", h: " << h << std::endl;
    std::cout << "Delta x: " << delta_x << std::endl;
    std::cout << "Delta y: " << delta_y << std::endl;

    unsigned *table = new unsigned[w * h];
    unsigned block_size = mandelbrot::get_block_size(w, h, n_procs - 1);

    for (unsigned i = 1; i < n_procs; ++i){
        int *bound = mandelbrot::get_boundaries(w, h, i, n_procs - 1, block_size);
        int start_y = bound[2], end_y = bound[3];
        unsigned local_h = end_y - start_y + 1;
        unsigned local_table_size = w * local_h;
        unsigned offset = block_size * w * (i - 1);

        MPI_Recv(
            table + offset,
            local_table_size,
            MPI_UNSIGNED,
            i,
            0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
        
    png::image< png::rgb_pixel > image = create_image(w,h,table);
    image.write(args.output_path);

    delete[] table;
}


void slave_work(
    params args,
    int rank, int n_procs,
    REAL_TYPE delta_x, REAL_TYPE delta_y,
    unsigned m
){
    d_mandelbrot(
        args.ex, args.n_threads,
        rank, n_procs - 1,
        args.c0, args.c1,
        delta_x, delta_y,
        args.w, args.h, m
    );
}

int main(int argc, char **argv){
    using COMPLEX::complex;
    using std::function;

    int error;

    // MPI Initialization
    // ----------------------------------------------
    if (error = MPI_Init(&argc, &argv)){
        std::cout << "Error in MPI_Init: " << error << std::endl;
    }


    int world_size, world_rank;
    if (error = MPI_Comm_size(MPI_COMM_WORLD, &world_size)){
        std::cout << "Error in MPI_Comm_size: " << error << std::endl;
    }

    if (error = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank)){
        std::cout << "Error in MPI_Comm_rank: " << error << std::endl;
    }
    // ----------------------------------------------

    params args = parse_args(argc, argv);

    unsigned w = args.w, h = args.h, m = 250;

    complex<REAL_TYPE> c0(args.c0),c1(args.c1);

    REAL_TYPE delta_x = (c1.real() - c0.real()) / w;
    REAL_TYPE delta_y = (c1.imag() - c0.imag()) / h;

    function <void(unsigned)>dummy = [](unsigned){};

    if(world_size > h + 1){
        function<void(unsigned)> proc_number_err_fn = function<void(unsigned)>(
            [](unsigned h){
                std::cerr << "Number of processes cannot be higher than image height + 1!" << std::endl;
                std::cerr << "Setting number of processes to h + 1 = " << h + 1 << std::endl;
            }
        );

        do_master(
            world_rank,
            proc_number_err_fn,
            dummy, h
        );

        world_size = h + 1; 
    }

    world_size = world_size > h + 1 ? h + 1 : world_size;

    if (world_size == 1){
        function<void(unsigned)> proc_number_err_fn = function<void(unsigned)>(
            [](unsigned h){
                std::cerr << "Number of processes needs to be at least 2! Exiting..." << std::endl;
            }
        );

        do_master(
            world_rank,
            proc_number_err_fn,
            dummy, h
        );


        if (error = MPI_Finalize()){
            std::cout << "Error in MPI_Finalize: " << error << std::endl;
        }

        exit(1);
    }

    if (world_rank > world_size - 1){
        if (error = MPI_Finalize()){
            std::cout << "Error in MPI_Finalize: " << error << std::endl;
        }
        exit(0);
    }
    
    std::function<void(
        params, int, int,
        REAL_TYPE, REAL_TYPE, unsigned
    )> slave_fn = slave_work, master_fn = master_work;

    // do master and slave main work depending on rank
    do_master(
        world_rank,
        master_fn,
        slave_fn,
        args, world_rank, world_size,
        delta_x, delta_y, m
    );

    if (error = MPI_Finalize()){
        std::cout << "Error in MPI_Finalize: " << error << std::endl;
    }

    return 0;
}
