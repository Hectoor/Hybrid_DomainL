#!/usr/bin/bash
set -ex
ls
export PYTHONPATH=`pwd`

CUDA_VISIBLE_DEVICES=3 python ../HDL_trainG2C.py --snapshot-dir /media/HDD_4TB/zyh/HDL/g2c_hds --restore-from /media/HDD_4TB/zyh/GTA5_init.pth --num-steps 200000 --num-steps-stop 200000
