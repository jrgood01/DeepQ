
#ifndef IMAGE_UTIL_H
#define IMAGE_UTIL_H

#include <iostream>
#include <cstdint>
#include <SDL.h>

void SaveImage(const std::string& filename, unsigned char* image_data, int width, int height);
std::shared_ptr<unsigned char[]> ProcessImage(const Uint32* src, int srcWidth, int srcHeight, int dstWidth, int dstHeight);



#endif // IMAGE_UTIL_H