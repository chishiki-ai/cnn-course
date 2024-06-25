#!/bin/bash
LOCAL_RANK=$PMIX_RANK

CMD="torch_train_distributed.py $@"

NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE


GPU_PER_NODE=4

if [[ -z "${NODEFILE}" ]]; then
    RANKS=$NODEFILE
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

PRELOAD="module reset;"
PRELOAD+="module load gcc/9.1.0 python3/3.9.2;"
PRELOAD+="source /scratch1/07980/sli4/training/cnn_course/bin/activate;"
PRELOAD+="export OMP_NUM_THREADS=8; "


LAUNCHER="python -m torch.distributed.launch "
LAUNCHER+="--nnodes=$NNODES  --nproc_per_node=$GPU_PER_NODE \
--node_rank=$LOCAL_RANK --master_addr=$MAIN_RANK --max_restarts=0 "

# Combine preload, launcher, and script+args into full command
FULL_CMD="$PRELOAD $LAUNCHER $CMD"

echo $FULL_CMD 

eval $FULL_CMD &

wait