#!/bin/bash
# Run EVA02-CLIP-B-16 evaluation. Set DATA_ROOT and EXP_DIR if needed. Run from evaluate/.

EVAL_ROOT=$(cd "$(dirname "$0")" && pwd)
NUM_TASKS=$(nvidia-smi -L 2>/dev/null | wc -l)
RUN_NAME=EVA02-CLIP-B-16
EXP_DIR=${EXP_DIR:-exp/cache}
mkdir -p "$EVAL_ROOT/$EXP_DIR/$RUN_NAME/logs"
cd "$EVAL_ROOT/models/${RUN_NAME}" || exit 1
echo "Detected $NUM_TASKS GPUs. Launching $NUM_TASKS parallel tasks."

for (( ID=0; ID<NUM_TASKS; ID++ )); do
    echo "Launching task $ID on GPU $ID"
    CUDA_VISIBLE_DEVICES=$ID nohup python ${RUN_NAME}_test.py \
        --calc_pass --num_chunks $NUM_TASKS --chunk_idx $ID --interval 30 \
        > "$EVAL_ROOT/$EXP_DIR/$RUN_NAME/logs/log_$ID.txt" 2>&1 &
done
echo "Tasks started. Use 'wait' to wait for completion."
