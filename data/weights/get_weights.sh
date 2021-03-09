#! /bin/bash

SHELL_FOLDER=$(dirname $(readlink -f "$0"))
cd SHELL_FOLDER
wget https://pjreddie.com/media/files/yolov3.weights