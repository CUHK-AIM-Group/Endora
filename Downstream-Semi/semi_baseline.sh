EXP_NAME="polypdiag_supervised_only"
DATASET="ucf101"
DATA_PATH="data/downstream/PolypDiag" # input the PolypDiag dataset path


CUDA_VISIBLE_DEVICES=1 python train_semi_baseline.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --epochs 20 \
  --master_port 29501 \
  --lr 0.001 \
  --batch_size_per_gpu 4 \
  --num_workers 4 \
  --num_labels 2 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/eval_data/$EXP_NAME" \
  --scratch \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/splits" \
  DATA.PATH_PREFIX "${DATA_PATH}/videos" \
  DATA.USE_FLOW False
  
