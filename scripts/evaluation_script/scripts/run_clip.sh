#!/bin/bash

# 自动获取 GPU 数量
NUM_TASKS=$(nvidia-smi -L | wc -l)
RUN_NAME=CLIP
mkdir -p exp/cache/$RUN_NAME/logs

cd models/${RUN_NAME}
pwd
# echo $PYTHONPATH

echo "Detected $NUM_TASKS GPUs"
echo "Launching $NUM_TASKS parallel tasks"

# 记得改LEFT
for (( ID=0; ID<$NUM_TASKS; ID++ ))
do
    GPU_ID=$ID
    echo "Launching task $ID on GPU $GPU_ID"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
        nohup python ${RUN_NAME}_test.py \
        --encode_video --num_chunks $NUM_TASKS --chunk_idx $ID --interval 10 \
        > /mnt/public/***/***/LOVR/exp/cache/$RUN_NAME/logs/log_$ID.txt 2>&1 &
done

# wait   # 等待所有任务完成

# srun -p mineru2_data -J test --gres=gpu:8 bash run_video_clip_xl.sh

# #!/bin/bash

# # 自动获取 GPU 数量
# NUM_TASKS=$(nvidia-smi -L | wc -l)
# RUN_NAME=VideoCLIP-XL
# mkdir -p exp/cache/$RUN_NAME/logs

# # 获取总 CPU 核数，每个任务均分
# TOTAL_CORES=$(nproc)
# CORES_PER_TASK=$((TOTAL_CORES / NUM_TASKS))

# echo "Detected $NUM_TASKS GPUs and $TOTAL_CORES CPU cores"
# echo "Each task will use $CORES_PER_TASK cores"

# for (( ID=0; ID<$NUM_TASKS; ID++ ))
# do
#     GPU_ID=$ID
#     CORE_START=$((ID * CORES_PER_TASK))
#     CORE_END=$((CORE_START + CORES_PER_TASK - 1))
#     CPU_RANGE="${CORE_START}-${CORE_END}"

#     echo "Launching task $ID on GPU $GPU_ID, CPU cores $CPU_RANGE"

#     CUDA_VISIBLE_DEVICES=$GPU_ID taskset -c $CPU_RANGE \
#         python $RUN_NAME_test.py --encode_text --num_chunks $NUM_TASKS --chunk_idx $ID --interval 30 \
#         > exp/cache/$RUN_NAME/logs/log_$ID.txt 2>&1 &
# done

# wait     # 等待所有任务完成

# # srun -p mineru2_data -J test --gres=gpu:8 bash run_video_clip_xl.sh

