import os
import json
import torch
import ffmpeg
import argparse, math, cv2, logging
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

debug = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def llm_init(model_path):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1, # for 7B model
        # tensor_parallel_size=4, # for 72B model
        trust_remote_code=True,
    )
    return llm

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    print(len(lst))
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def encode(x):
    return f'{x["vid"]}-{x["slice_num"]}'

def resize_video_to_240p(video_path, file):
    global debug
    result_path = f'' # your result file path
    if debug:
        ffmpeg.input(video_path).output(
            result_path, vf="scale=426x240"
        ).overwrite_output().run()
    else:
        ffmpeg.input(video_path).output(
            result_path, vf="scale=426x240"
        ).overwrite_output().global_args('-loglevel', 'quiet').run()
    return result_path

def main(args):
    if not args.debug:
        global debug
        debug = False
        logging.getLogger("vllm").setLevel(logging.WARNING)
    
    with open(args.jsonl_file, 'r') as f:
        infos = [json.loads(line) for line in f.readlines()]
    infos = get_chunk(infos, args.num_chunks, args.chunk_idx)
    
    result_file = args.result_file
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    if args.rerun:
        with open(result_file, 'w') as f:
            pass
    if os.path.exists(result_file):
        processed = []
        with open(result_file, 'r') as f:
            processed = [json.loads(line) for line in f.readlines()]
        req = set([encode(i) for i in infos])
        with open(result_file, 'w') as f:
            st = set()
            for i in processed:
                if encode(i) in st or encode(i) not in req:
                    continue
                st.add(encode(i))
                f.write(json.dumps(i) + '\n')
        with open(result_file, 'r') as f:
            processed = [json.loads(line) for line in f.readlines()]
        processed = set([encode(data) for data in processed])
        infos = [i for i in infos if encode(i) not in processed]

    llm = llm_init(args.model_path)
    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=1024,
        stop_token_ids=[],
    )

    processor = AutoProcessor.from_pretrained(args.model_path)

    f = open(result_file, 'a')
    process_list = []
    
    print(f'{len(infos)} items to process')
    
    for i, info in enumerate(tqdm(infos)):
        vpath = os.path.join(args.video_folder, info['path'])
        process_list.append([i, vpath])
        if len(process_list) < args.batch_size and info != infos[-1]:
            continue
        llm_inputs = []
        for j, data in enumerate(process_list):
            idx, vpath = data[0], data[1]

            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                tqdm.write(f'error when open {vpath}\n')
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 1:
                print(f'video {vpath} is 1 frame video')
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            if frame_count < 5:
                print(f'video {vpath} has less than 5 frames')
                continue
            duration = int(frame_count / fps) if fps > 0 else 0
            fnum = min(64, max(8, duration))
            if frame_count < 8:
                fnum = 2

            messages = [
                {
                    "role": "system",
                    "content": "You are a video analysis assistant, and your task is to provide a detailed and structured description of the provided video."
                }, {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{vpath}",
                            "nframes": fnum,
                        }, {
                            "type": "text",
                            "text": "Please provide a detailed and comprehensive description of the video content. \
                                Your description should cover the events, scenes, character actions, overall style, \
                                emotional atmosphere, narrative elements, and any noteworthy details. \
                                Ensure that your description is clear, coherent, and thorough, with as much detail as possible."
                        },
                    ],
                },
            ]
            
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            mm_data = {}
            if image_inputs is not None: mm_data["image"] = image_inputs
            if video_inputs is not None: mm_data["video"] = video_inputs
            
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            })
        
        generate_num = 1
        
        for generate_index in range(generate_num):
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            
            for j, data in enumerate(process_list):
                idx, vpath = data[0], data[1]
                max_retries = 2
                generated_text = "" 

                try:
                    generated_text = outputs[j].outputs[0].text
                except (IndexError, AttributeError) as e:
                    print(f"Attempt failed")
                    generated_text = ""
                        
                infos[idx]['cap'] = generated_text
        
        for j, data in enumerate(process_list):
            idx, vpath = data[0], data[1]
            rinfo = infos[idx]
            f.write(json.dumps(rinfo) + '\n')
        
        f.flush()
        process_list = []

    f.close()
    
    assert len(process_list) == 0

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"Current CUDA Device Index: {torch.cuda.current_device()}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-path", default='your model path')
    parser.add_argument("--video-folder", default='your video folder')
    parser.add_argument("--jsonl-file", default='your jsonl file')
    parser.add_argument("--result-file", default='your result file')
    parser.add_argument("--num-chunks", type=int, default=2)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    main(args)
