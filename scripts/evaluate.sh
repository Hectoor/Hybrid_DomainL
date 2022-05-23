set -ex
ls
export PYTHONPATH=`pwd`


CUDA_VISIBLE_DEVICES=1 python evaluate.py \
        --exp-root='/mnt/mdisk/zyh/asanet/asanethdl/hdl/'
