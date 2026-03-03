#include <iostream>
#include <armadillo>
#include <raylib.h>
#include "kernel.hpp"
#include <fftw3.h>
#include "grid.hpp"
#include "Lenia.hpp"
#include <chrono>
#include <thread>


arma::Mat<float> ring_matrix(int N, double inner_r, double outer_r) {
    arma::Mat<float> M(N, N, arma::fill::zeros);
    double cx = (N - 1) / 2.0;
    double cy = (N - 1) / 2.0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double dist = std::sqrt(std::pow(i - cx, 2) + std::pow(j - cy, 2));
            if (dist >= inner_r && dist <= outer_r)
                M(i, j) = 1;
        }
    }
    return M;
}


void print_ring(const arma::Mat<float>& M) {
    for (arma::uword i = 0; i < M.n_rows; ++i) {
        for (arma::uword j = 0; j < M.n_cols; ++j)
            std::cout << (M(i, j) ? "█" : "·") << " ";
        std::cout << "\n";
    }
}







int main(){

    InitWindow(800, 800, "Lenia Simulation");
    SetTargetFPS(10);

    int N = 10;
    double inner_r = 3.0;
    double outer_r = 5.0;

    
   if(!fftwf_init_threads()){
        std::cerr << "Thread init failed\n" ;
        return 1;
   }
   fftwf_plan_with_nthreads(std::thread::hardware_concurrency());



   
   Lenia lenia(800, 800, "Gaussian - 0.0, 0.15", 10);

   

   arma::Mat<float> R = ring_matrix(N, inner_r, outer_r);

   arma::Mat<float> initialState = arma::repmat(R, 80, 80);

   lenia.setInitBoardState(initialState);
   lenia.Start();

   while(WindowShouldClose() == false){

     lenia.updateBoardState();


     BeginDrawing();

     ClearBackground(GRAY);

     lenia.renderWorldState();

     EndDrawing();

   }

   lenia.destroyWorld();
   CloseWindow();








}




