#!/bin/bash
# We need this proxy so not to put the shebang into `slurm_job.py`
# We cannot put a shebang there since we use different python executors for it

module load cuda
module load gcc/10.2.0

CUDA_LAUNCH_BLOCKING=1 python $1 # start train


