#!/bin/sh
TOOLS=../../build/tools
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/opt/OpenBLAS/lib:/data/libs/cudnn-7.0/lib64:/usr/local/lib

$TOOLS/caffe train -solver solver.prototxt -gpu 0 -weights ../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
echo 'Done.'
