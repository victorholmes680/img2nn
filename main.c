#include <stdio.h>
#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

#if define(__APPLE__)
    #define STB_IMAGE_IMPLEMENTATION
    #define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image.h"
#include "stb_image_write.h"

#include <raylib.h>
#include <raymath.h>
#include <assert.h>

// define global variables
size_t arch[] = {3, 11, 11, 11, 11, 11, 1};
size_t max_epoch = 100*1000;
size_t batches_per_frame = 200;
size_t batch_size = 28;
float rate = 1.0f;
float scroll = 0.f;
bool paused = true;

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

    // block:initialize the neural network
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(nn, -1, 1);
    // endblock

    // initialize the windows size
    size_t WINDOW_FACTOR = 80;
    size_t WINDOW_WIDTH = (16*WINDOW_FACTOR);
    size_t WINDOW_HEIGHT = (9*WINDOW_FACTOR);

    // initialize the properties of windows
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "img2nn");
    SetTargetFPS(60);

    // region preivew the output we generated
    Gym_Plot plot = {0};
    Font font = LoadFontEx("./font/iosevka-regular.ttf", 72, NULL, 0);
    SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);

    size_t preview_width = 28;
    size_t preview_height = 28;

    Image preview_image1 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture1 = LoadTextureFromImage(preview_image1);

    Image preview_image2 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture2 = LoadTextureFromImage(preview_image2);

    Image preview_image3 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D preview_texture3 = LoadTextureFromImage(preview_image3);
    // endregion

    
    // region: draw two original images
    Image original_image1 = GenImageColor(img1_width, img1_height, BLACK);
    for(size_t y = 0; y < (size_t) img1_height; ++y) {
        for(size_t x = 0; x < (size_t) img1_width; ++x) {
            uint8_t pixel = img1_pixels[u*img1_width + x];
            ImageDrawPixel(&original_image1, x, y, CLITERAL(Color) {pixel, pixel, pixel, 255});
        }
    }
    Texture2D original_texture1 = LoadTextureFromImage(original_image1);
    
    Image original_image2 = GenImageColor(img2_width, img2_height, BLACK);
    for(size_t y = 0; y < (size_t) img2_height; ++y) {
        for(size_t x = 0; x < (size_t) img2_width; ++x) {
            uint8_t pixel = img2_pixels[u*img2_width + x];
            ImageDrawPixel(&original_image2, x, y, CLITERAL(Color) {pixel, pixel, pixel, 255});
        }
    }
    Texture2D original_texture2 = LoadTextureFromImage(original_image2);
    // endregion


    size_t out_width = 512;
    size_t out_height = 512;
    uint8_t *out_pixels = malloc(sizeof(*out_pixels)*out_width*out_height);
    assert(out_pixels != NULL);

    Gym_Batch gb = {0};
    bool rate_dragging = false;
    bool scroll_dragging = false;
    size_t epoch = 0;

    while(!WindowShouldClose()) {
        if(IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if(IsKeyPressed(KEY_R)) {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }

        // save the merged image by scrolling rate
        if(IsKeyPressed(KEY_S)) {
            for(size_t y = 0; y < out_height; ++y) {
                for(size_t x = 0; x < out_width; ++x) {
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(out_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)x/(out_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 2) = scroll;
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
                    out_pixels[y*out_width + x] = pixel;
                }
            }
            const char *out_file_path = "merged_scaled.png";
            if(!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width*sizeof(*out_pixels))) {
                fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
                return 1;
            }
            printf("Generated %s from %s\n", out_file_path, img1_file_path);
        }

        // region: update the parameters of neural network in batches
        for(size_t i = 0; i < batches_per_frame && !paused && epoch < max_epoch; ++i) {
            gym_process_batch(&gb, batch_size, nn, g, t, rate);
            if(gb.finished) {
                epoch += 1;
                da_append(&plot, gb.cost);
                mat_shuffle_rows(t);
            }
        }
        // endregion

        
        
    }
    
    
    
    
    return 0;

    
}
