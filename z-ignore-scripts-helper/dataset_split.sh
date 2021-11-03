#!/bin/bash

# dataset path should not end in /
dataset_name=$1
dataset_train=${dataset_name}-train
dataset_val=${dataset_name}-val
echo 'actual dataset_path :'
echo $dataset_name
echo 'validation dataset path :'
echo $dataset_val


# Make directories to split dataset into train and valid
mkdir -p -v $dataset_val
mkdir -p -v "$dataset_val"/resized-files/preprocessed-camera-normals/rgb-visualizations
mkdir -p -v "$dataset_val"/resized-files/preprocessed-outlines/rgb-visualizations
mkdir -p -v "$dataset_val"/resized-files/preprocessed-rgb-imgs
mkdir -p -v "$dataset_val"/source-files/camera-normals/rgb-visualizations
mkdir -p -v "$dataset_val"/source-files/component-masks
mkdir -p -v "$dataset_val"/source-files/depth-imgs
mkdir -p -v "$dataset_val"/source-files/depth-imgs-rectified
mkdir -p -v "$dataset_val"/source-files/json-files
mkdir -p -v "$dataset_val"/source-files/outlines/rgb-visualizations
mkdir -p -v "$dataset_val"/source-files/rgb-imgs
mkdir -p -v "$dataset_val"/source-files/variant-masks
mkdir -p -v "$dataset_val"/source-files/world-normals
mkdir -p -v "$dataset_val"/source-files/segmentation-masks

# Make directories to split dataset into train and valid
mkdir -p -v $dataset_train
mkdir -p -v "$dataset_train"/resized-files/preprocessed-camera-normals/rgb-visualizations
mkdir -p -v "$dataset_train"/resized-files/preprocessed-outlines/rgb-visualizations
mkdir -p -v "$dataset_train"/resized-files/preprocessed-rgb-imgs
mkdir -p -v "$dataset_train"/source-files/camera-normals/rgb-visualizations
mkdir -p -v "$dataset_train"/source-files/component-masks
mkdir -p -v "$dataset_train"/source-files/depth-imgs
mkdir -p -v "$dataset_train"/source-files/depth-imgs-rectified
mkdir -p -v "$dataset_train"/source-files/json-files
mkdir -p -v "$dataset_train"/source-files/outlines/rgb-visualizations
mkdir -p -v "$dataset_train"/source-files/rgb-imgs
mkdir -p -v "$dataset_train"/source-files/variant-masks
mkdir -p -v "$dataset_train"/source-files/world-normals
mkdir -p -v "$dataset_train"/source-files/segmentation-masks

# To determine the percentage split
count=`ls -ltr ${dataset_name}/resized-files/preprocessed-rgb-imgs/* | wc -l`
last_count=$((count-1))
count=$((last_count/10))
echo $count

echo 'moving files to val dataset '
# splitting resized files
for i in $(eval echo "{000000000..$count}");
do
cp "$dataset_name"/resized-files/preprocessed-camera-normals/$i-cameraNormals.exr "$dataset_val"/resized-files/preprocessed-camera-normals/
cp "$dataset_name"/resized-files/preprocessed-camera-normals/rgb-visualizations/$i-cameraNormals.png "$dataset_val"/resized-files/preprocessed-camera-normals/rgb-visualizations/
cp "$dataset_name"/resized-files/preprocessed-outlines/$i-outlineSegmentation.png "$dataset_val"/resized-files/preprocessed-outlines/
cp "$dataset_name"/resized-files/preprocessed-outlines/rgb-visualizations/$i-outlineSegmentation.png "$dataset_val"/resized-files/preprocessed-outlines/rgb-visualizations/
cp "$dataset_name"/resized-files/preprocessed-rgb-imgs/$i-rgb.png "$dataset_val"/resized-files/preprocessed-rgb-imgs/

# splitting source files
cp "$dataset_name"/source-files/camera-normals/$i-cameraNormals.exr "$dataset_val"/source-files/camera-normals/
cp "$dataset_name"/source-files/camera-normals/rgb-visualizations/$i-cameraNormals.png "$dataset_val"/source-files/camera-normals/rgb-visualizations/
cp "$dataset_name"/source-files/component-masks/$i-componentMasks.exr "$dataset_val"/source-files/component-masks/
cp "$dataset_name"/source-files/depth-imgs/$i-depth.exr "$dataset_val"/source-files/depth-imgs/
cp "$dataset_name"/source-files/depth-imgs-rectified/$i-depth-rectified.exr "$dataset_val"/source-files/depth-imgs-rectified/
cp "$dataset_name"/source-files/json-files/$i-masks.json "$dataset_val"/source-files/json-files/
cp "$dataset_name"/source-files/outlines/$i-outlineSegmentation.png "$dataset_val"/source-files/outlines/
cp "$dataset_name"/source-files/outlines/rgb-visualizations/$i-outlineSegmentationRgb.png "$dataset_val"/source-files/outlines/rgb-visualizations/
cp "$dataset_name"/source-files/rgb-imgs/$i-rgb.jpg "$dataset_val"/source-files/rgb-imgs/
cp "$dataset_name"/source-files/variant-masks/$i-variantMasks.exr "$dataset_val"/source-files/variant-masks/
cp "$dataset_name"/source-files/world-normals/$i-normals.exr "$dataset_val"/source-files/world-normals/
cp "$dataset_name"/source-files/segmentation-masks/$i-segmentation-mask.png "$dataset_val"/source-files/segmentation-masks/

done
echo 'moved files to val directory '
echo 'moving files to train dataset'
for i in $(seq -f "%09g" $count $last_count);
# for i in $(eval echo  "$(seq  )");
do
cp "$dataset_name"/resized-files/preprocessed-camera-normals/$i-cameraNormals.exr "$dataset_train"/resized-files/preprocessed-camera-normals/
cp "$dataset_name"/resized-files/preprocessed-camera-normals/rgb-visualizations/$i-cameraNormals.png "$dataset_train"/resized-files/preprocessed-camera-normals/rgb-visualizations/
cp "$dataset_name"/resized-files/preprocessed-outlines/$i-outlineSegmentation.png "$dataset_train"/resized-files/preprocessed-outlines/
cp "$dataset_name"/resized-files/preprocessed-outlines/rgb-visualizations/$i-outlineSegmentation.png "$dataset_train"/resized-files/preprocessed-outlines/rgb-visualizations/
cp "$dataset_name"/resized-files/preprocessed-rgb-imgs/$i-rgb.png "$dataset_train"/resized-files/preprocessed-rgb-imgs/

# splitting source files
cp "$dataset_name"/source-files/camera-normals/$i-cameraNormals.exr "$dataset_train"/source-files/camera-normals/
cp "$dataset_name"/source-files/camera-normals/rgb-visualizations/$i-cameraNormals.png "$dataset_train"/source-files/camera-normals/rgb-visualizations/
cp "$dataset_name"/source-files/component-masks/$i-componentMasks.exr "$dataset_train"/source-files/component-masks/
cp "$dataset_name"/source-files/depth-imgs/$i-depth.exr "$dataset_train"/source-files/depth-imgs/
cp "$dataset_name"/source-files/depth-imgs-rectified/$i-depth-rectified.exr "$dataset_train"/source-files/depth-imgs-rectified/
cp "$dataset_name"/source-files/json-files/$i-masks.json "$dataset_train"/source-files/json-files/
cp "$dataset_name"/source-files/outlines/$i-outlineSegmentation.png "$dataset_train"/source-files/outlines/
cp "$dataset_name"/source-files/outlines/rgb-visualizations/$i-outlineSegmentationRgb.png "$dataset_train"/source-files/outlines/rgb-visualizations/
cp "$dataset_name"/source-files/rgb-imgs/$i-rgb.jpg "$dataset_train"/source-files/rgb-imgs/
cp "$dataset_name"/source-files/variant-masks/$i-variantMasks.exr "$dataset_train"/source-files/variant-masks/
cp "$dataset_name"/source-files/world-normals/$i-normals.exr "$dataset_train"/source-files/world-normals/
cp "$dataset_name"/source-files/segmentation-masks/$i-segmentation-mask.png "$dataset_train"/source-files/segmentation-masks/

done
echo 'moved files to train directory '

