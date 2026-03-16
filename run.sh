#!/bin/sh
set -xe
CFLAGS="-I./thirdparty/"
cc $CFLAGS -o main main.c && ./main ./img1.png ./img2.png
