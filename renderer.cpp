#include "renderer.hpp"
#include <armadillo>


Renderer::Renderer(int rows, int columns) : rows(rows), columns(columns){

    Image img = GenImageColor(rows, columns, BLACK);
    texture = LoadTextureFromImage(img);
    UnloadImage(img);
    
    pixels = new Color[rows * columns];

}

void Renderer::DestroyMatrixRenderer(){
    UnloadTexture(texture);
    delete[] pixels;
}

void Renderer::DrawMatrix(){
    DrawTextureEx(texture, {0.0, 0.0}, 0.0,  1.0, WHITE);
}

void Renderer::UpdateMatrixRenderer(arma::Mat<float>& worldState){
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < columns; c++){
            float v = worldState(r, c);
            unsigned char g = (unsigned char)(v * 255);
            pixels[r * columns + c] = Color{g, g, g, 255};
        }
    }

    UpdateTexture(texture, pixels);
}
