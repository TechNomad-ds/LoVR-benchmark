import os
import json
import argparse
import math
from multiprocessing import Pool, Manager, Process
from tqdm import tqdm
from openai import OpenAI
import requests
import time
from datetime import datetime


API_URL = "" # Replace with your actual API URL
API_KEY = "" # Replace with your actual API key
BASE_PATH = "" # Replace with your actual base path
PROGRESS_LOG = "" # Replace with your actual progress log path

def log_progress(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    with open(PROGRESS_LOG, "a") as f:
        f.write(log_message)
        f.flush()


def openai_chat(messages, model='gpt-4o', max_retry=3):
    headers = {
        'Authorization': f"Bearer {API_KEY}",
        'Content-Type': 'application/json',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
    }
    
    for attempt in range(max_retry):
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful video summarizer."},
                    {"role": "user", "content": messages}
                ],
                "temperature": 0.5
            }
            response = requests.post(API_URL, headers=headers, json=payload, timeout=1800)
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"API returned non-200 status code: {response.status_code}, attempt {attempt + 1}/{max_retry}")
                
        except requests.exceptions.Timeout:
            print(f"API timeout (attempt {attempt + 1}/{max_retry})")
        except Exception as e:
            print(f"API error: {type(e).__name__} (attempt {attempt + 1}/{max_retry})")
    
    return None


def process_video_chunk(args, vid_chunk, progress_dict, result_queue, caps_data, worker_id):
    results = []
    total_vids = len(vid_chunk)
    
    processed_clusters = set()
    if os.path.exists(args.result_file):
        with open(args.result_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data['vid'] in vid_chunk:
                        processed_clusters.add(data['cluster_id'])
                except json.JSONDecodeError:
                    continue
    
    for vid in vid_chunk:
        try:                
            caps = [cap for cap in caps_data if cap['vid'] == vid]
            caps_sorted = sorted(caps, key=lambda x: x['slice_num'])
            
            if not caps_sorted:
                continue
            
            clusters = []
            for i in range(0, len(caps_sorted), 10):
                cluster_clips = caps_sorted[i:i+10]
                if not cluster_clips:
                    continue
                
                current_cluster_id = f"{vid}_{i//10}"
                if current_cluster_id in processed_clusters:
                    continue
                
                merged_paragraphs = []
                
                first_clip = cluster_clips[0]
                first_paragraphs = [p.strip() for p in first_clip['cap'].split('\n') if p.strip()]
                merged_paragraphs.extend(first_paragraphs)
                
                for j in range(1, len(cluster_clips)):
                    current_clip = cluster_clips[j]
                    
                    if not isinstance(current_clip, dict) or 'cap' not in current_clip:
                        log_progress(f"Worker {worker_id} - Invalid clip format in {vid} cluster {i//10}")
                        break
                    
                    current_paragraphs = [p.strip() for p in current_clip['cap'].split('\n') if p.strip()]
                    if not current_paragraphs:
                        continue
                    
                    prev_last_para = merged_paragraphs[-1] if merged_paragraphs else ""
                    curr_first_para = current_paragraphs[0] if current_paragraphs else ""
                    
                    if not prev_last_para or not curr_first_para:
                        merged_paragraphs.extend(current_paragraphs)
                        continue
                    
                    prompt = prompt = f"""Strictly follow these requirements for caption connection. Given the last paragraph of the previous caption and the first paragraph of the next caption, optimize the connection between them.:

                    # Original Content
                    [Last paragraph of previous caption]:
                    {prev_last_para}

                    [First paragraph of next paragraph]:
                    {curr_first_para}

                    # Requirements
                    1. Preserve facts: Keep all key information including names, actions, and objects unchanged
                    2. Transition optimization:
                    - Add temporal transition words (e.g., "then", "after that") to connect the two paragraphs
                    - Slightly adjust sentence order to improve logical flow
                    3. Conciseness: Remove repetitive content and keep language concise
                    4. Format specifications:
                    - There must be a \n between the paragraphs
                    - Do not include any additional explanations or notes, only output the modified paragraphs

                    # Output format (must strictly follow, Do not include any additional explanations or notes):
                    [Optimized end of previous paragraph] \n [Optimized beginning of next paragraph]
                    """
                    
                    optimized = openai_chat(prompt)
                    if "[Optimized end of previous paragraph]" in optimized:
                        optimized = optimized.replace("[Optimized end of previous paragraph]", "").strip()
                    if "[Optimized beginning of next paragraph]" in optimized:
                        optimized = optimized.replace("[Optimized beginning of next paragraph]", "").strip()
                    
                    if optimized:
                        try:
                            if '\n' in optimized:
                                parts = optimized.split('\n', 1)
                                new_prev = parts[0].strip()
                                new_curr = parts[1].strip()
                                
                                merged_paragraphs[-1] = new_prev
                                current_paragraphs[0] = new_curr
                            else:
                                new_prev = optimized.strip()
                                new_curr = ""
                                merged_paragraphs[-1] = new_prev
                                current_paragraphs[0] = new_curr
                                
                        except Exception as e:
                            log_progress(f"Worker {worker_id} - Error parsing optimized text: {str(e)}")
                    
                    merged_paragraphs.extend(current_paragraphs)
                
                final_cap = '\n'.join(merged_paragraphs)
                
                clusters.append({
                    'vid': vid,
                    'start_slice_num': cluster_clips[0]['slice_num'],
                    'end_slice_num': cluster_clips[-1]['slice_num'],
                    'cap': final_cap,
                    'clip_count': len(cluster_clips),
                    'cluster_id': current_cluster_id,
                    'merged_paragraphs': len(merged_paragraphs)
                })
            
            if clusters:
                results.extend(clusters)
                for cluster in clusters:
                    result_queue.put(cluster)
            
            progress = len(progress_dict) / total_vids * 100
            log_progress(f"Worker {worker_id} progress: {len(progress_dict)}/{total_vids} ({progress:.2f}%)")
        
        except Exception as e:
            log_progress(f"Worker {worker_id} error processing video {vid}: {str(e)}")
            continue
    
    return results

def writer_process(result_file, result_queue):
    with open(result_file, 'a') as f:
        while True:
            result = result_queue.get()
            if result == "DONE":
                break
            f.write(json.dumps(result) + '\n')
            f.flush()

def main(args):
    with open(PROGRESS_LOG, "w") as f:
        f.write(f"Process started at {datetime.now()}\n")
        f.write(f"Total workers: {args.num_workers}\n")
        f.flush()
    
    log_progress("Starting to load caption data...")
    with open(args.cap_file, 'r') as f:
        all_caps = [json.loads(line) for line in f]
    
    vids = {cap['vid'] for cap in all_caps}
    vids = list(vids)
    total_vids = len(vids)
    log_progress(f"Loaded {len(all_caps)} captions for {total_vids} videos")
    
    processed = set()
    if os.path.exists(args.result_file):
        log_progress("Checking previously processed videos...")
        with open(args.result_file, 'r') as f:
            processed_vids = {json.loads(line)['vid'] for line in f}
            
            cluster_counts = {}
            f.seek(0)
            for line in f:
                data = json.loads(line)
                cluster_counts[data['vid']] = cluster_counts.get(data['vid'], 0) + 1
            
            f.seek(0)
            expected_counts = {}
            for vid in processed_vids:
                clip_count = sum(1 for item in all_caps if item['vid'] == vid)
                expected_counts[vid] = (clip_count + 9) // 10
            
            processed = {vid for vid in processed_vids 
                        if cluster_counts.get(vid, 0) == expected_counts.get(vid, 0)}
        
        log_progress(f"Found {len(processed)} already processed videos")
    
    remaining_vids = [vid for vid in vids if vid not in processed]
    log_progress(f"Remaining videos to process: {len(remaining_vids)}")
    
    vid_chunks = [remaining_vids[i::args.num_workers] for i in range(args.num_workers)]
    log_progress(f"Split videos into {len(vid_chunks)} chunks for parallel processing")
    
    caps_chunks = []
    for chunk in vid_chunks:
        chunk_vids = set(chunk)
        caps_chunks.append([cap for cap in all_caps if cap['vid'] in chunk_vids])
    
    manager = Manager()
    progress_dict = manager.dict({vid: True for vid in processed})
    result_queue = manager.Queue()
    
    log_progress("Starting writer process...")
    writer = Process(target=writer_process, args=(args.result_file, result_queue))
    writer.start()
    
    log_progress("Starting worker processes...")
    start_time = time.time()
    
    with Pool(args.num_workers) as pool:
        tasks = [(args, chunk, progress_dict, result_queue, caps, i) 
                for i, (chunk, caps) in enumerate(zip(vid_chunks, caps_chunks))]
        pool.starmap(process_video_chunk, tasks)
    
    result_queue.put("DONE")
    writer.join()
    
    duration = time.time() - start_time
    log_progress(f"All workers completed in {duration:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-file", default='your cap file')
    parser.add_argument("--result-file", default='your result file')
    parser.add_argument("--num-workers", type=int, default=50)
    args = parser.parse_args()
    
    main(args)