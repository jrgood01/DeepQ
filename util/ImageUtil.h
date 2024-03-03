
#ifndef IMAGE_UTIL_H
#define IMAGE_UTIL_H

#include <iostream>
#include <cstdint>
#include <SDL.h>

void SaveImage(const std::string& filename, unsigned char* image_data, int width, int height);
void ProcessImage(const uint32_t* src, unsigned char* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight);



#endif // IMAGE_UTIL_H