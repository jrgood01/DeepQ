#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "ImageUtil.h"
#include <SDL.h>
void SaveImage(const std::string& filename, unsigned char* image_data, int width, int height) {    
    // Assuming the image has 4 channels (RGBA)
    int channels = 1;

    int result = stbi_write_png(filename.c_str(), width, height, channels, image_data, width * channels);

    if(result){
        std::cout << "Image saved successfully.\n";
    }else{
        std::cout << "Failed to save image.\n";
    }
}

void ProcessImage(const Uint32* src, unsigned char* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    float widthRatio = (float)srcWidth / dstWidth;
    float heightRatio = (float)srcHeight / dstHeight;

    for (int y = 0; y < dstHeight; ++y) {
        for (int x = 0; x < dstWidth; ++x) {
            int srcX = (int)(x * widthRatio);
            int srcY = (int)(y * heightRatio);

            // Average the pixel values in the source image to downscale
            Uint32 sumR = 0;
            Uint32 sumG = 0;
            Uint32 sumB = 0;
            int count = 0;
            for (int dy = 0; dy < heightRatio; ++dy) {
                for (int dx = 0; dx < widthRatio; ++dx) {
                    Uint32 pixel = src[(srcY + dy) * srcWidth + (srcX + dx)];
                    Uint32 r = (pixel >> 16) & 0xFF;
                    Uint32 g = (pixel >> 8) & 0xFF;
                    Uint32 b = pixel & 0xFF;
                    sumR += r;
                    sumG += g;
                    sumB += b;
                    ++count;
                }
            }
            Uint32 avgR = sumR / count;
            Uint32 avgG = sumG / count;
            Uint32 avgB = sumB / count;
            unsigned char gray = (avgR + avgG + avgB) / 3;

            // Store the grayscale value in the destination image
            dst[y * dstWidth + x] = gray;
        }
    }
}