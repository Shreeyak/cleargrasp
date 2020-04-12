#!/usr/bin/env bash

bin/x86_64/depth2depth \
 "sample_files/input-depth.png" \
 "sample_files/output-depth.png" \
 -xres 256 -yres 144 \
 -fx 185 -fy 185 \
 -cx 128 -cy 72 \
 -inertia_weight 1000 \
 -smoothness_weight 0.001 \
 -tangent_weight 1 \
 -input_normals "sample_files/normals.h5" \
 -input_tangent_weight "sample_files/occlusion-weight.png"


python convert_intermediate_data_to_rgb.py --sample_files_dir "sample_files/"
