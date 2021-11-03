#!/bin/bash

# # deleting depth rectified images
# echo 'deleting depth rectified images'
# rm -rf scoop-val/source-files/depth-imgs-rectified
# rm -rf scoop-train/source-files/depth-imgs-rectified
# rm -rf sphere-bath-bomb-val/source-files/depth-imgs-rectified
# rm -rf sphere-bath-bomb-train/source-files/depth-imgs-rectified
# rm -rf tree-bath-bomb-val/source-files/depth-imgs-rectified
# rm -rf tree-bath-bomb-train/source-files/depth-imgs-rectified
# rm -rf star-bath-bomb-val/source-files/depth-imgs-rectified
# rm -rf star-bath-bomb-train/source-files/depth-imgs-rectified
# rm -rf short-bottle-no-cap-val/source-files/depth-imgs-rectified
# rm -rf short-bottle-no-cap-train/source-files/depth-imgs-rectified
# rm -rf test-tube-with-cap-val/source-files/depth-imgs-rectified
# rm -rf test-tube-with-cap-train/source-files/depth-imgs-rectified
# rm -rf test-tube-no-cap-val/source-files/depth-imgs-rectified
# rm -rf test-tube-no-cap-train/source-files/depth-imgs-rectified
# rm -rf cup-with-waves-val/source-files/depth-imgs-rectified
# rm -rf cup-with-waves-train/source-files/depth-imgs-rectified
# rm -rf heart-bath-bomb-val/source-files/depth-imgs-rectified
# rm -rf heart-bath-bomb-train/source-files/depth-imgs-rectified
# rm -rf stemless-champagne-glass-val/source-files/depth-imgs-rectified
# rm -rf stemless-champagne-glass-train/source-files/depth-imgs-rectified
# rm -rf flower-bath-bomb-val/source-files/depth-imgs-rectified
# rm -rf flower-bath-bomb-train/source-files/depth-imgs-rectified
# rm -rf short-bottle-with-cap-val/source-files/depth-imgs-rectified
# rm -rf short-bottle-with-cap-train/source-files/depth-imgs-rectified

# # deleting outlines files
# echo 'deleting outlines files'
# rm -rf scoop-val/source-files/outlines
# rm -rf scoop-train/source-files/outlines
# rm -rf sphere-bath-bomb-val/source-files/outlines
# rm -rf sphere-bath-bomb-train/source-files/outlines
# rm -rf tree-bath-bomb-val/source-files/outlines
# rm -rf tree-bath-bomb-train/source-files/outlines
# rm -rf star-bath-bomb-val/source-files/outlines
# rm -rf star-bath-bomb-train/source-files/outlines
# rm -rf short-bottle-no-cap-val/source-files/outlines
# rm -rf short-bottle-no-cap-train/source-files/outlines
# rm -rf test-tube-with-cap-val/source-files/outlines
# rm -rf test-tube-with-cap-train/source-files/outlines
# rm -rf test-tube-no-cap-val/source-files/outlines
# rm -rf test-tube-no-cap-train/source-files/outlines
# rm -rf cup-with-waves-val/source-files/outlines
# rm -rf cup-with-waves-train/source-files/outlines
# rm -rf heart-bath-bomb-val/source-files/outlines
# rm -rf heart-bath-bomb-train/source-files/outlines
# rm -rf stemless-champagne-glass-val/source-files/outlines
# rm -rf stemless-champagne-glass-train/source-files/outlines
# rm -rf flower-bath-bomb-val/source-files/outlines
# rm -rf flower-bath-bomb-train/source-files/outlines
# rm -rf short-bottle-with-cap-val/source-files/outlines
# rm -rf short-bottle-with-cap-train/source-files/outlines

# # deleting preprocessed outlines files
# echo 'deleting preprocessed outlines files'
# rm -rf scoop-val/resized-files/preprocessed-outlines
# rm -rf scoop-train/resized-files/preprocessed-outlines
# rm -rf sphere-bath-bomb-val/resized-files/preprocessed-outlines
# rm -rf sphere-bath-bomb-train/resized-files/preprocessed-outlines
# rm -rf tree-bath-bomb-val/resized-files/preprocessed-outlines
# rm -rf tree-bath-bomb-train/resized-files/preprocessed-outlines
# rm -rf star-bath-bomb-val/resized-files/preprocessed-outlines
# rm -rf star-bath-bomb-train/resized-files/preprocessed-outlines
# rm -rf short-bottle-no-cap-val/resized-files/preprocessed-outlines
# rm -rf short-bottle-no-cap-train/resized-files/preprocessed-outlines
# rm -rf test-tube-with-cap-val/resized-files/preprocessed-outlines
# rm -rf test-tube-with-cap-train/resized-files/preprocessed-outlines
# rm -rf test-tube-no-cap-val/resized-files/preprocessed-outlines
# rm -rf test-tube-no-cap-train/resized-files/preprocessed-outlines
# rm -rf cup-with-waves-val/resized-files/preprocessed-outlines
# rm -rf cup-with-waves-train/resized-files/preprocessed-outlines
# rm -rf heart-bath-bomb-val/resized-files/preprocessed-outlines
# rm -rf heart-bath-bomb-train/resized-files/preprocessed-outlines
# rm -rf stemless-champagne-glass-val/resized-files/preprocessed-outlines
# rm -rf stemless-champagne-glass-train/resized-files/preprocessed-outlines
# rm -rf flower-bath-bomb-val/resized-files/preprocessed-outlines
# rm -rf flower-bath-bomb-train/resized-files/preprocessed-outlines
# rm -rf short-bottle-with-cap-val/resized-files/preprocessed-outlines
# rm -rf short-bottle-with-cap-train/resized-files/preprocessed-outlines

# # running data pre-processing script
# echo 'running pre-processing script'

# cd ../../utils

# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/sccop-val --num_start 0
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/sccop-train --num_start 49
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/sphere-bath-bomb-val --num_start 0
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/sphere-bath-bomb-train --num_start 49
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/tree-bath-bomb-val --num_start 0
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/tree-bath-bomb-train --num_start 49
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/star-bath-bomb-val --num_start 0
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/star-bath-bomb-train --num_start 49
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/test-tube-with-cap-val --num_start 0
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/test-tube-with-cap-train --num_start 49
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/test-tube-no-cap-val --num_start 0
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/test-tube-no-cap-train --num_start 49
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/cup-with-waves-val --num_start 0
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/cup-with-waves-train --num_start 49
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/heart-bath-bomb-val --num_start 0
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/heart-bath-bomb-train --num_start 49
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/stemless-champagne-glass-val --num_start 0
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/stemless-champagne-glass-train --num_start 49
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/flower-bath-bomb-val --num_start 0
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/flower-bath-bomb-train --num_start 49
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/short-bottle-with-cap-val --num_start 0
# python3.5 data_processing_script.py --p ../data/datasets/empt --root ../data/datasets/short-bottle-with-cap-train --num_start 49


# compresiing files to move it into storage
tar -czvf scoop-val.tar.gz scoop-val
tar -czvf scoop-train.tar.gz scoop-train
tar -czvf sphere-bath-bomb-val.tar.gz sphere-bath-bomb-val
tar -czvf sphere-bath-bomb-train.tar.gz sphere-bath-bomb-train
tar -czvf tree-bath-bomb-val.tar.gz tree-bath-bomb-val
tar -czvf tree-bath-bomb-train.tar.gz tree-bath-bomb-train
tar -czvf star-bath-bomb-val.tar.gz star-bath-bomb-val
tar -czvf star-bath-bomb-train.tar.gz star-bath-bomb-train
tar -czvf short-bottle-no-cap-val.tar.gz short-bottle-no-cap-val
tar -czvf short-bottle-no-cap-train.tar.gz short-bottle-no-cap-train
tar -czvf test-tube-with-cap-val.tar.gz test-tube-with-cap-val
tar -czvf test-tube-with-cap-train.tar.gz test-tube-with-cap-train
tar -czvf test-tube-no-cap-val.tar.gz test-tube-no-cap-val
tar -czvf test-tube-no-cap-train.tar.gz test-tube-no-cap-train
tar -czvf cup-with-waves-val.tar.gz cup-with-waves-val
tar -czvf cup-with-waves-train.tar.gz cup-with-waves-train
tar -czvf heart-bath-bomb-val.tar.gz heart-bath-bomb-val
tar -czvf heart-bath-bomb-train.tar.gz heart-bath-bomb-train
tar -czvf stemless-champagne-glass-val.tar.gz stemless-champagne-glass-val
tar -czvf stemless-champagne-glass-train.tar.gz stemless-champagne-glass-train
tar -czvf flower-bath-bomb-val.tar.gz flower-bath-bomb-val
tar -czvf flower-bath-bomb-train.tar.gz flower-bath-bomb-train
tar -czvf short-bottle-with-cap-val.tar.gz short-bottle-with-cap-val
tar -czvf short-bottle-with-cap-train.tar.gz short-bottle-with-cap-train
tar -czvf milk-bottles-val.tar.gz milk-bottles-val
tar -czvf milk-bottles-train.tar.gz milk-bottles-train
tar -czvf milk-bottles-in-boxes-val.tar.gz milk-bottles-in-boxes-val
tar -czvf milk-bottles-in-boxes-train.tar.gz milk-bottles-in-boxes-train


# moving datasets to google cloud storage
gsutil -m cp cup-with-waves-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp cup-with-waves-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp flower-bath-bomb-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp flower-bath-bomb-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp heart-bath-bomb-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp heart-bath-bomb-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp scoop-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp scoop-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp short-bottle-no-cap-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp short-bottle-no-cap-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp short-bottle-with-cap-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp short-bottle-with-cap-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp sphere-bath-bomb-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp sphere-bath-bomb-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp star-bath-bomb-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp star-bath-bomb-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp stemless-champagne-glass-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp stemless-champagne-glass-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp test-tube-no-cap-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp test-tube-no-cap-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp test-tube-with-cap-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp test-tube-with-cap-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp tree-bath-bomb-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp tree-bath-bomb-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp milk-bottles-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp milk-bottles-train.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp milk-bottles-in-boxes-val.tar.gz gs://greppy-gbrain/transparent/datasets/
gsutil -m cp milk-bottles-in-boxes-train.tar.gz gs://greppy-gbrain/transparent/datasets/

