#pragma once
#include <armadillo>
#include <fftw3.h>


arma::Mat<float> bell(arma::Mat<float> input, float mean, float std);



class KernelShape{
    protected:
        int rows;
        int columns;

    public:
        KernelShape(int r): rows(r), columns(r){}

        virtual ~KernelShape() = default;

        int getRows() const {return rows;}
        int getColumns() const {return columns;}
 
        virtual arma::Mat<float> generate() const = 0;
};


class GaussianKernel : public KernelShape{

    private:
        float peak;
        float sigma;
    
    public:
        GaussianKernel(int r, float p, float s) : KernelShape(r), peak(p), sigma(s){}

        arma::Mat<float> generate() const override;
};

/*

class RingKernel : public KernelShape{

    private:
        float peak;
        float sigma;
        int R;
    
    public:
        RingKernel(int r, float p, float s) : KernelShape(r), peak(p), sigma(s), R(r){}

        arma::Mat<float> generate() const override{
            arma::fvec v = arma::linspace<arma::fvec>(-R, R - 1, 2*R);

            arma::fmat X = arma::repmat(v.t(), 2*R, 1);
            arma::fmat Y = arma::repmat(v, 1, 2*R);

            arma::fmat D = arma::sqrt(X % X + Y % Y) / R;

            arma::fmat mask = arma::conv_to<arma::fmat>::from(D < 1.0f);
        

            arma::fmat result = mask % bell(D, peak, sigma);

            float s = arma::accu(result);
            if (s > 0) result /= s;

            return result;
        }
};


*/

class KernelManager{

    private:
        int worldRows;
        int worldColumns;

        arma::Mat<float> rawKernel = {
        {0,0,0,0,1,1,1,0,0,0,0},
        {0,0,1,1,1,1,1,1,1,0,0},
        {0,1,1,1,1,1,1,1,1,1,0},
        {0,1,1,1,1,1,1,1,1,1,0},
        {1,1,1,1,0,0,0,1,1,1,1},
        {1,1,1,1,0,0,0,1,1,1,1},
        {1,1,1,1,0,0,0,1,1,1,1},
        {0,1,1,1,1,1,1,1,1,1,0},
        {0,1,1,1,1,1,1,1,1,1,0},
        {0,0,1,1,1,1,1,1,1,0,0},
        {0,0,0,0,1,1,1,0,0,0,0}
        };

        arma::Mat<float> ballKernel = rawKernel / arma::accu(rawKernel);



        std::unordered_map<std::string, arma::Mat<arma::cx_float>> kernelsDict;




    public:
        KernelManager(int worldRows, int worldColumns) : worldRows(worldRows), worldColumns(worldColumns){}

        void buildKernel(const std::string& name, const KernelShape& shape);  // For when there needs to be a Gaussian Type kernel.
        void buildKernel(const std::string& name, const arma::Mat<float>& kernel);  // For when there is an input armadillo matrix that needs to function as a kernel.

        void initDefaultKernels();
        
        const arma::Mat<arma::cx_float>& getKernel(const std::string& name) const{
            return kernelsDict.at(name);
        }

        



};






