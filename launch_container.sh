# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

VID_DIR=$1
OUT_DIR=$2

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

sudo screen docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --network=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$VID_DIR,dst=/video,type=bind,readonly \
    --mount src=$OUT_DIR,dst=/output,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src linjieli222/hero-video-feature-extractor:mil-nce \
    bash -c "source /src/setup.sh && bash" \
