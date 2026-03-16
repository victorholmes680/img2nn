#include <stdio.h>
#include "nn.h"
#include "stb_image.h"
#include "stb_image_write.h"
// extract the first argument from the input stream
// and forward the pointer to next
char *args_shift(int *argc, char ***argv) {
    NN_ASSERT(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

int main(int argc, char **argv)
{
    printf("Hello, Cherno!\n");
    const char *program = args_shift(&argc, &argv);
    printf("the program name is %s\n", program);

    // make sure the remaining count of argument is larget than zero
    if(argc <= 0) {
        fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
        fprintf(stderr, "ERROR: no image1 is provided\n");
        return 1;
    }

    const char *img1_file_path = args_shift(&argc, &argv);
    printf("image1 file path is %s\n", img1_file_path);
    if(argc <= 0) {
        fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
        fprintf(stderr, "ERROR: no image2 is provided\n");
        return 1;
    }

    const char *img2_file_path = args_shift(&argc, &argv);
    printf("image2 file path is %s\n", img2_file_path);

    // at this point, the command is mathing the format what we want
    int img1_width, img1_height, img1_channels;
    // the last parameter represent using default converting type
    uint8_t *img1_pixels = (uint8_t *)stbi_load(img1_file_path, &img1_width, &img1_height, &img_channels, 0);
    if(img1_pixels == NULL) {
        fprintf(stderr, "ERROR: could not read image %s\n", img1_file_path);
        return 1;
    }
    if(img1_channels != 1) {
        fprintf(stderr, "ERROR: %s is %d bits images. Only 8 bit grayscale images are supported\n", img1_file_path, img1_channels*8);
        return 1;
    }

    int img2_width, img2_height, img2_channels;
    // the last parameter represent using default converting type
    uint8_t *img2_pixels = (uint8_t *)stbi_load(img2_file_path, &img2_width, &img2_height, &img2channels, 0);
    if(img2_pixels == NULL) {
        fprintf(stderr, "ERROR: could not read image %s\n", img2_file_path);
        return 1;
    }
    if(img2_channels != 1) {
        fprintf(stderr, "ERROR: %s is %d bits images. Only 8 bit grayscale images are supported\n", img2_file_path, img2_channels*8);
        return 1;
    }

    printf("%s size %dx%d %d bits\n", img1_file_path, img1_width, img1_height, img1_channels*8);
    printf("%s size %dx%d %d bits\n", img2_file_path, img2_width, img2_height, img2_channels*8);

    //==============================================================================================================
    // complete dealing with input files
      
    // this is the training data, row represent the number of samples, and column means input and output parameters
    Mat t = mat_alloc(img1_width*img1_height + img2_width*img2_height, NN_INPUT(nn).cols + NN_OUTPUT(nn).cols);

    // region: initialize the training data from the two images program read
    for(size_t y = 0; y < (size_t)img1_height; ++y) {
        for(size_t x = 0; x < (size_t)img1_width; ++x) {
            size_t i = y * img1_width + x;
            MAT_AT(t, i, 0) = (float)x/(img1_width - 1);
            MAT_AT(t, i, 1) = (float)y/(img1_width - 1);
            // position at column = 2 mean which picture is current pixel belong to 
            MAT_AT(t, i, 2) = 0.f;

            // make thegray value be the range from 0 to 1
            MAT_AT(t, i, 3) = img1_pixels[i]/255.f;
        }
    }

    for(size_t y = 0; y < (size_t)img2_height; ++y) {
        for(size_t x = 0; x < (size_t)img2_width; ++x) {
            size_t i = y * img2_width + x;
            MAT_AT(t, i, 0) = (float)x/(img2_width - 1);
            MAT_AT(t, i, 1) = (float)y/(img2_width - 1);
            // position at column = 2 mean which picture is current pixel belong to
            MAT_AT(t, i, 2) = 1.f;

            // make thegray value be the range from 0 to 1
            MAT_AT(t, i, 3) = img2_pixels[i]/255.f;
        }
    }
    // endregion

    // make the sequence of the row of martix unordered
    mat_shuffle_rows(t);
    
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(nn, -1, 1);
    
    return 0;
}
