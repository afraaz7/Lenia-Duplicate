#include "kernel.hpp"



arma::Mat<float> GaussianKernel::generate() const{
    arma::Mat<float> K(rows, columns);

    int center_c = rows / 2;
    int center_r = columns / 2;

    for(int i = 0; i < rows; ++i){
        for (int j = 0; j < columns; ++j){
                    
            float dx = i - center_r;
            float dy = j - center_c;
            float r = std::sqrt(dx * dx + dy * dy);

            K(i, j) = std::exp(-std::pow(r - peak, 2) / (2.0f * sigma * sigma));


            }
    }

    K /= arma::accu(K);

    return K;

}







arma::Mat<float> bell(arma::Mat<float> input, float mean, float std){
    arma::Mat<float> result = arma::exp(-(arma::pow(((input - mean) / std), 2)) / 2);
    return result;
}
    

void KernelManager::buildKernel(const std::string& name, const KernelShape& shape){

    

    arma::Mat<float> skeletonKernel = shape.generate();

    int kernelrows = skeletonKernel.n_rows;
    int kernelcols = skeletonKernel.n_cols;

    int shift_r = kernelrows / 2;
    int shift_c = kernelcols / 2;

    arma::Mat<float> spatialkernel(worldRows, worldColumns, arma::fill::zeros);
    

    spatialkernel(arma::span(0, kernelrows - 1), arma::span(0, kernelcols - 1)) = skeletonKernel;

    // Shift the kernel so that the focus of the kernel, i.e the midpoint is at the origin.
    spatialkernel = arma::circshift(spatialkernel, -shift_c, 0);
    spatialkernel = arma::circshift(spatialkernel, -shift_r, 1);


    arma::Mat<arma::cx_float> F_Kernel = arma::fft2(spatialkernel);

    kernelsDict[name] = std::move(F_Kernel);

}

void KernelManager::buildKernel(const std::string& name, const arma::Mat<float>& kernel){

    //Assume that the Kernel matrix is already given and does not have to be generated.
    // In this case, the kernel just has to be shifted like in the previous case and added to the dictionary.

    int kernelrows = kernel.n_rows;
    int kernelcols = kernel.n_cols;

    int shift_r = kernelrows / 2;
    int shift_c = kernelcols / 2;
    
    arma::Mat<float> spatialKernel(worldRows, worldColumns, arma::fill::zeros);

    spatialKernel(arma::span(0, kernelrows - 1), arma::span(0, kernelcols - 1)) = kernel;

    spatialKernel = arma::circshift(spatialKernel, -shift_c, 0);
    spatialKernel = arma::circshift(spatialKernel, -shift_r, 1);

    arma::Mat<arma::cx_float> F_kernel = arma::fft2(spatialKernel);

    kernelsDict[name] = std::move(F_kernel);

}

void KernelManager::initDefaultKernels(){

    buildKernel("Gaussian - 0.0, 0.15", GaussianKernel(5, 0.0, 0.15));

    buildKernel("Gaussian - 0.0, 0.2", GaussianKernel(5, 0.0, 0.2));

    buildKernel("ballKernel", ballKernel);

}


