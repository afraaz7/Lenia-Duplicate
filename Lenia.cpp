#include "Lenia.hpp"


void Lenia::initializeFFTPlans(){

    plan = fftwf_plan_dft_2d(boardState_rows, 
                            boardState_columns, 
                            grid.getSpatialPointer(), 
                            grid.getFreqPointer(),
                            FFTW_FORWARD, FFTW_ESTIMATE);

    plan_reverse = fftwf_plan_dft_2d(boardState_rows,
                            boardState_columns,
                            reinterpret_cast<fftwf_complex*>(grid.getFreqPointer()),
                            reinterpret_cast<fftwf_complex*>(grid.getSpatialPointer()),
                            FFTW_BACKWARD, FFTW_ESTIMATE);

   

}


arma::Mat<float> Lenia::growthFunction(arma::Mat<float>& input, float mean, float std){

    arma::Mat<float> result = arma::exp(-(arma::pow(((input - mean) / std), 2)) / 2);
    return result;

}


void Lenia::setInitBoardState(arma::Mat<float>& initialState){
    grid.setRealWorld(initialState);
}



void Lenia::updateBoardState(){

    if(isRunning()){
        fftwf_execute(plan);

        grid.multiplyKernel(kernel);

        fftwf_execute(plan_reverse);

        arma::Mat<float> result = grid.generateRenderableState();

        result += (1.0 / timescale) * (growthFunction(result, 0.135, 0.015));

        result.clamp(0.0, 1.0);

        renderer.UpdateMatrixRenderer(result);

        grid.setRealWorld(result);
    
    }
}

void Lenia::renderWorldState(){
    if(isRunning()){
        renderer.DrawMatrix();
    }
}

void Lenia::destroyWorld(){

    renderer.DestroyMatrixRenderer();
    fftwf_destroy_plan(plan);
    fftwf_destroy_plan(plan_reverse);
    fftwf_cleanup();

}