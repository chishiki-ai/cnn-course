SOURCE_TAR=$1
DEST_DIR=$2

mkdir -p $DEST_DIR

FULL_CMD="cp -r $SOURCE_TAR $DEST_DIR ; "
FULL_CMD+="tar zxf $DEST_DIR/data.tar.gz -C $DEST_DIR; "
FULL_CMD+="ls $DEST_DIR/Dataset_2; "
FULL_CMD+="rm $DEST_DIR/data.tar.gz; "


NODEFILE=/tmp/nodelist
scontrol show hostnames $SLURM_NODELIST > $NODEFILE

if [[ -z "${NODEFILE}" ]]; then
    RANKS=$HOSTNAME
else
    RANKS=$(tr '\n' ' ' < $NODEFILE)
fi

echo "Command: $FULL_CMD"

# Launch execute the command on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait