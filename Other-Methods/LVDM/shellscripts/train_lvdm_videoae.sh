

PROJ_ROOT="./log"                        # root directory for saving experiment logs
EXPNAME="lvdm_videoae_cholec"          # experiment name 
DATADIR="/mnt/zhen_chen/Dora/data/CholecT45_frames"    # dataset directory

CONFIG="configs/videoae/cholec.yaml"

# run
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=7 python main.py \
--base $CONFIG \
-t --gpus 0, \
--name $EXPNAME \
--logdir $PROJ_ROOT \
--auto_resume True \
lightning.trainer.num_nodes=1 \
data.params.train.params.data_root=$DATADIR \
model.params.ddconfig.encoder.first_stage_config.params.ckpt_path=$AEPATH

# -------------------------------------------------------------------------------------------------
# commands for multi nodes training
# - use torch.distributed.run to launch main.py
# - set `gpus` and `lightning.trainer.num_nodes`

# For example:

# python -m torch.distributed.run \
#     --nproc_per_node=8 --nnodes=$NHOST --master_addr=$MASTER_ADDR --master_port=1234 --node_rank=$INDEX \
#     main.py \
#     --base $CONFIG \
#     -t --gpus 0,1,2,3,4,5,6,7 \
#     --name $EXPNAME \
#     --logdir $PROJ_ROOT \
#     --auto_resume True \
#     lightning.trainer.num_nodes=$NHOST \
#     data.params.train.params.data_root=$DATADIR \
#     data.params.validation.params.data_root=$DATADIR
