#!/usr/bin/env bash
# wujian@2019

set -eu

epoches=100
batch_size=32
cache_size=8
chunk_size=64256
eval_interval=3000

echo "$0 $@"

[ $# -ne 2 ] && echo "Script format error: $0 <exp-id> <gpu-id>" && exit 1

exp_id=$1
gpu_id=$2

./nnet/train.py \
  --gpu $gpu_id \
  --checkpoint exp/nnet/$exp_id \
  --batch-size $batch_size \
  --cache-size $cache_size \
  --chunk-size $chunk_size \
  --epoches $epoches \
  --eval-interval $eval_interval \
  > $exp_id.train.log 2>&1
