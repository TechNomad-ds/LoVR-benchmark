# LoVR Benchmark

## 📁 Dataset Generation Code

This project consists of three core steps. The code is located in the `scripts/` folder.

1. **Clip Segmentation**
2. **Caption Generation**
3. **Merge Captions**

Please follow the order above when executing the steps.

### 1. Video Segmentation (`clip_segmentation.py`)

This script splits videos in the input folder into smaller clips based on certain rules and saves them to a specified directory.

Parameters:

- `--input_folder`: Path to the folder containing original video files  
- `--output_dir`: Output directory for the segmented video clips  
- `--max_workers`: Maximum number of threads for concurrent processing  

Example Command:

```bash
python clip_segmentation.py \
    --input_folder /path/to/your/videos \
    --output_dir /path/to/output/clips \
    --max_workers 50
```

### 2. Caption Generation (`caption_generator.py`)

This script generates captions for the video clips produced in the previous step.

Parameters:

- `--model-path`: Path to the model weights file  
- `--video-folder`: Directory containing the video clips from the previous step  
- `--jsonl-file`: Input JSONL file recording video clip information  
- `--result-file`: Output file path for results (generated per chunk)  
- `--batch-size`: Batch size used during inference  
- `--num-chunks`: Number of chunks to split the task into  
- `--chunk-idx`: Index of the current chunk being processed (starting from 0)  
- `--LOG_FILE`: Log output file path (optional)  

Example Command (Chunked Processing):

```bash
export CKPT=/path/to/model_weights
export BASE=/path/to/workdir
CHUNKS=8
IDX=0
LOG_FILE=output_log_${IDX}.log

python caption_generator.py \
    --model-path ${CKPT} \
    --video-folder your/video/folder \
    --jsonl-file your/jsonl/file \
    --result-file your/result/file \
    --batch-size 16 \
    --num-chunks ${CHUNKS} \
    --chunk-idx ${IDX} \
    > "$LOG_FILE" 2>&1 &
```

### 3. Merge Caption Results (`caption_merger.py`)

This script merges all generated caption files from individual chunks into a single final output file in JSONL format.

Parameters:

- `--cap-file`: Output file path after merging  
- `--result-file`: Paths to all chunk result files (wildcard matching supported)  
- `--num-workers`: Number of workers for parallel processing  

Example Command:

```bash
python caption_merger.py \
    --cap-file /path/to/final_caption_output.jsonl \
    --result-file "/path/to/results/result_0_*.jsonl" \
    --num-workers 50
```

## 4. Evaluation Code and Scripts

The evaluation pipeline is located in the `evaluation_script/` directory:

- `evaluation_script/code/`: Contains model-specific evaluation implementations.
- `evaluation_script/scripts/`: Shell scripts to run evaluations with predefined configurations.


## ⚠️ Responsible Use Policy

We encourage responsible usage of the LoVR benchmark. Users should not use the dataset to develop harmful, discriminatory, or privacy-invasive applications. We recommend performing fairness audits and adhering to ethical AI principles when using this dataset.

If you make use of this dataset in your work, please cite our paper (link coming soon).
