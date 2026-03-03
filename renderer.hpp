#pragma once


#include <raylib.h>
#include <armadillo>

class Renderer{

    public:
        Renderer(int rows, int columns);

        void DestroyMatrixRenderer();
        void DrawMatrix();
        void UpdateMatrixRenderer(arma::Mat<float>& worldState);


    private:
        int rows;
        int columns;
        Texture2D texture;
        Color* pixels;
};

