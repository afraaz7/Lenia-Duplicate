#pragma once
#include "kernel.hpp"
#include <armadillo>
#include "grid.hpp"
#include <fftw3.h>
#include "renderer.hpp"


class Lenia{

    private:
        KernelManager kernelmanager;
        int boardState_columns;
        int boardState_rows;
        Grid grid;
        Renderer renderer;

        arma::Mat<arma::cx_float> kernel;
        bool run;
        int timescale;

        fftwf_plan plan;
        fftwf_plan plan_reverse;

    public:

        Lenia(int rows, int columns, const std::string& kernelName, int timescale) : 
                                        boardState_rows(rows), 
                                        boardState_columns(columns), 
                                        grid(rows, columns), 
                                        renderer(rows, columns),
                                        timescale(timescale),
                                        run(false),
                                        kernelmanager(rows, columns){
                                            kernelmanager.initDefaultKernels();
                                            kernel = kernelmanager.getKernel(kernelName);
                                        };

        void Start(){run = true;}
        void Stop(){run = false;}

        void initializeFFTPlans();

        arma::Mat<float> growthFunction(arma::Mat<float>& input, float mean, float std);

        void setInitBoardState(arma::Mat<float>& initialState);

        void updateBoardState();

        bool isRunning(){return run;}

        void renderWorldState();

        void destroyWorld();
        






};