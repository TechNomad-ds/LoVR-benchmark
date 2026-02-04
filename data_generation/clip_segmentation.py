import os
import concurrent.futures
from tqdm import tqdm
import logging
import argparse
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def split_video_into_clips(video_path, output_dir, base_threshold=34.0, min_clip_duration=5.0, min_clips=64, max_clips=100):
    os.makedirs(output_dir, exist_ok=True)

    while True:
        video_manager = VideoManager([video_path])
        video_manager.set_downscale_factor(3) 
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(
            threshold=base_threshold,
            min_scene_len=int(min_clip_duration * video_manager.get_framerate()) 
        ))

        try:
            video_manager.start()
            scene_manager.detect_scenes(video_manager)
            scene_list = scene_manager.get_scene_list()
            num_clips = len(scene_list)

            if True:
                split_video_ffmpeg([video_path], scene_list, output_dir)
                logging.info(f"Split into {num_clips} clips.")
                break
            else:
                base_threshold *= 0.9 if num_clips < min_clips else 1.1
                logging.info(f"Adjust threshold to {base_threshold:.2f}")

        except Exception as e:
            logging.error(f"Error: {e}")
            break
        finally:
            video_manager.release()


def process_video(video_path, output_dir, index):
    video_name = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_name)[0]
    output_subdir = os.path.join(output_dir, video_name_without_ext)

    if os.path.exists(output_subdir) and len(os.listdir(output_subdir)) > 0:
        logging.info(f"Skipping already processed video: {video_path}")
        return

    split_video_into_clips(video_path, output_subdir)



def process_videos_in_folder(input_folder, output_dir, max_workers):
    video_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.mp4', '.mkv', '.avi'))]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, video_path, output_dir, index): video_path 
                   for index, video_path in enumerate(video_files, start=1)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(video_files), desc="Processing videos"):
            video_path = futures[future]
            try:
                future.result(timeout=3600)  
            except Exception as e:
                logging.error(f"Error processing video {video_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos in a folder.")
    parser.add_argument("--input_folder", default="your_video_folder", help="Path to the folder containing videos.")
    parser.add_argument("--output_dir", default="your_output_clip_folder", help="Path to the output directory for clips.")
    parser.add_argument("--max_workers", type=int, default=50, help="Maximum number of threads.")
    args = parser.parse_args()

    try:
        process_videos_in_folder(args.input_folder, args.output_dir, args.max_workers)
    except KeyboardInterrupt:
        logging.info("Process interrupted. Cleaning up...")
