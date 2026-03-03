#pragma once

#include <armadillo>
#include <fftw3.h>

class Grid{
    public:
        Grid(int rows, int columns) : worldState_rows(rows), worldState_columns(columns),
                                    spatial_buffer(rows, columns, arma::fill::zeros),
                                    frequency_buffer(rows, columns, arma::fill::zeros){}


    arma::Mat<arma::cx_float>& getSpatialBuffer(){return spatial_buffer;}

    arma::Mat<arma::cx_float>& getFreqBuffer(){return frequency_buffer;}

    void multiplyKernel(arma::Mat<arma::cx_float>& kernel){frequency_buffer = frequency_buffer % kernel;}

    void setRealWorld(const arma::Mat<float>& realValues){
        spatial_buffer.zeros();
        spatial_buffer.set_real(realValues);
    }


    fftwf_complex* getSpatialPointer(){
        return reinterpret_cast<fftwf_complex*>(spatial_buffer.memptr());
    }

    fftwf_complex* getFreqPointer(){
        return reinterpret_cast<fftwf_complex*>(frequency_buffer.memptr());
    }

    arma::Mat<float> generateRenderableState(){
        arma::Mat<float> realPart = arma::real(spatial_buffer) / (worldState_rows * worldState_columns);
        return realPart;
    }

    private:
        int worldState_rows;
        int worldState_columns;
        arma::Mat<arma::cx_float> spatial_buffer;
        arma::Mat<arma::cx_float> frequency_buffer;


};

