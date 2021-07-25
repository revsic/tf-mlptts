# CUDA_VISIBLE_DEVICES=7 python train.py \
#     --data-dir /datasets/ljspeech \
#     --from-raw \
#     --name k193affl3_e200

# CUDA_VISIBLE_DEVICES=7 python train.py \
#     --config /hdd1/revsic/ckpt/mlptts/k193aff.json \
#     --load-epoch 199 \
#     --data-dir /datasets/ljspeech \
#     --from-raw \
#     --name k193aff

# CUDA_VISIBLE_DEVICES=0 python dump.py \
#     --reader ljspeech \
#     --data-dir /datasets/ljspeech \
#     --from-raw \
#     --target acoustic \
#     --path /datasets/ljspeech/acoustic

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --data-dir /datasets/ljspeech/tfrecord/acoustic.tfrecord \
#     --from-tfrecord \
#     --name recordtest \
#     --auto-rename
