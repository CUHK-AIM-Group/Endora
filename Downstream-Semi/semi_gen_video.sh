EXP_NAME="polypdiag_semi_fixmatch_Endora" 
DATASET="ucf101" 
DATA_PATH="data/downstream/PolypDiag" # input the PolypDiag dataset path
GEN_DATA_PATH="data/unlabeled/endora_colon" # input the path to the generated videos 


CUDA_VISIBLE_DEVICES=1 python eval_semi.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --epochs 20 \
  --master_port 29500 \
  --lr 0.001 \
  --batch_size_per_gpu 4 \
  --num_workers 0 \
  --num_labels 2 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/eval_data/$EXP_NAME" \
  --master_port "29501" \
  --scratch \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/splits" \
  DATA.PATH_PREFIX "${DATA_PATH}/videos" \
  DATA.PATH_TO_GEN_DATA_DIR "${GEN_DATA_PATH}/splits" \
  DATA.USE_FLOW False
