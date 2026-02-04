# LoVR Benchmark

## 📁 Dataset Generation Code

This project consists of three core steps. The code is located in the `data_generation/` folder.

1. **Clip Segmentation**
2. **Caption Generation**
3. **Merge Captions**

Please follow the order above when executing the steps.

### 1. Clip Segmentation (`clip_segmentation.py`)

This script splits videos in the input folder into smaller clips based on certain rules and saves them to a specified directory.

Parameters:

- `--input_folder`: Path to the folder containing original video files  
- `--output_dir`: Output directory for the segmented video clips  
- `--max_workers`: Maximum number of threads for concurrent processing  

Example Command:

```bash
cd data_generation
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
- `--rerun`: (optional) Rerun even if result already exists  
- `--debug`: (optional) Enable debug mode  

Example Command (Chunked Processing):

```bash
cd data_generation
export CKPT=/path/to/model_weights
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

This script merges caption data and writes the final output to a single JSONL file. It supports resuming: already processed videos in the result file will be skipped.

Parameters:

- `--cap-file`: Input JSONL file containing all caption data  
- `--result-file`: Output file path for the merged result (also used for resume)  
- `--num-workers`: Number of workers for parallel processing  

Example Command:

```bash
cd data_generation
python caption_merger.py \
    --cap-file /path/to/caption_data.jsonl \
    --result-file /path/to/final_merged_output.jsonl \
    --num-workers 50
```

---

## 📊 Evaluation Code and Scripts

The evaluation pipeline is located in the `evaluate/` directory.

- **`evaluate/models/`** – Model-specific evaluation implementations (e.g. CLIP, SigLIP, VideoCLIP-XL, LanguageBind, MM-Embed).
- **`evaluate/run_*.sh`** – Shell scripts to run evaluations with predefined configurations (e.g. `run_LanguageBind.sh`, `run_jina-clip-v2.sh`).

Evaluation supports **text-to-video** and **video-to-text** retrieval and computes **pass@k** metrics. For full usage, pipeline stages, and how to add a custom model, see **[evaluate/README.md](evaluate/README.md)**.

Pre-trained weights for supported models are available at **ModelScope**: `thirstylearning/lovr_models`. Download and extract them under `evaluate/models/` so that each model resides in its own subdirectory.

---

## ⚠️ Responsible Use Policy

We encourage responsible usage of the LoVR benchmark. Users should not use the dataset to develop harmful, discriminatory, or privacy-invasive applications. We recommend performing fairness audits and adhering to ethical AI principles when using this dataset.

If you make use of this dataset in your work, please cite our paper (link coming soon).
