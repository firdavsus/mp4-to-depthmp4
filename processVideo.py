import os
import cv2
import time
import shutil
import itertools
import subprocess
import numpy as np
from tqdm import tqdm
import multiprocessing
import concurrent.futures
from collections import deque
from scipy.ndimage import gaussian_filter

def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

def split_video_into_frames(input_video):
    output_folder = "./out/color_frames/"

    # Clear the output folder
    clear_folder(output_folder)

    # Call FFmpeg on the command line to split the video into PNGs
    cmd = f"ffmpeg -i {input_video} {output_folder}/frame_%07d.png"
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error running FFmpeg: {e}")
    return output_folder

def generate_depth_from_frames(color_frames_folder):
    dest_folder = "./input/"
    output_folder = "./out/depth_frames/"
    depth_generation_command = "python run.py --model_type dpt_beit_large_512 --input_path input --output_path output"
    depth_generation_output_folder = "./output/"

    # Clear dest_folder
    clear_folder(dest_folder)

    # Move color frames into dest_folder
    for filename in os.listdir(color_frames_folder):
        shutil.copy(os.path.join(color_frames_folder, filename), dest_folder)

    # Run depthGenerationCommand on the command line and wait for it to finish
    subprocess.run(depth_generation_command, shell=True, check=True)

    # Clear output_folder
    clear_folder(output_folder)

    # Move files from depth_generation_output_folder to output_folder
    for filename in os.listdir(depth_generation_output_folder):
        if filename.endswith(".png"):
            src_path = os.path.join(depth_generation_output_folder, filename)
            dst_path = os.path.join(output_folder, filename)
            shutil.move(src_path, dst_path)

    return output_folder

def apply_bilateral_filter(frame, d=5, sigma_color=75, sigma_space=75):
    frame_32f = frame.astype(np.float32)
    filtered_frame = cv2.bilateralFilter(frame_32f, d, sigma_color, sigma_space)
    return filtered_frame.astype(np.uint8)

def read_frame(depth_frames_folder, file_name):
    return cv2.imread(os.path.join(depth_frames_folder, file_name), cv2.IMREAD_UNCHANGED)

def apply_gaussian(frame, sigma):
    return gaussian_filter(frame, sigma=sigma, order=0, mode='reflect')

def calculate_difference(frame1, frame2):
    return np.abs(frame1.astype(np.int32) - frame2.astype(np.int32))

def detect_camera_cut(frame_threshold, threshold):
    return frame_threshold > threshold

def update_taa_buffer(curr_frame, taa_buffer, frame_threshold, dynamic_threshold, camera_cut_mult = 0.85):
    if frame_threshold > dynamic_threshold:
        return (taa_buffer * (1.0 - camera_cut_mult)) + curr_frame * camera_cut_mult; 
    else:
        return (curr_frame + taa_buffer) / 2.0

def apply_TAA(depth_frames_folder, n=5, kernel_size=3, sigma=1, threshold_mult=1.05):
    file_list = sorted(os.listdir(depth_frames_folder))
    num_files = len(file_list)

    # Initialize the TAA buffer with the first frame
    taa_buffer = read_frame(depth_frames_folder, file_list[0])

    # Get the target size from the first frame
    target_size = (taa_buffer.shape[1], taa_buffer.shape[0])

    # Initialize the rolling average frame threshold
    rolling_average_frame_threshold = 0.0

    # Apply TAA to each subsequent frame and save the result to disk
    for i in tqdm(range(1, num_files), desc="Applying TAA"):
        # Read the current frame
        curr_frame = read_frame(depth_frames_folder, file_list[i])

        # Resize the current frame to the target size
        curr_frame = cv2.resize(curr_frame, target_size)

        # Apply Gaussian smoothing to the current frame
        curr_frame_smooth = apply_gaussian(curr_frame, sigma)

        # Calculate the difference between the current frame and the TAA buffer
        frame_difference = calculate_difference(curr_frame_smooth, taa_buffer)

        # Update the rolling average frame threshold
        frame_threshold = np.mean(frame_difference) / 255
        rolling_average_frame_threshold = (
            rolling_average_frame_threshold * (i - 1) + frame_threshold
        ) / i

        # Calculate the dynamic threshold
        dynamic_threshold = rolling_average_frame_threshold * threshold_mult

        # Update the TAA buffer
        taa_buffer = update_taa_buffer(curr_frame_smooth, taa_buffer, frame_threshold, dynamic_threshold)

        # Normalize pixel values
        max_val = np.max(taa_buffer)
        output_frame = taa_buffer / max_val * 255

        # Apply Bilateral smoothing to the output frame
        output_frame = apply_bilateral_filter(output_frame)

        # Save the output frame to disk
        output_file = os.path.join(depth_frames_folder, file_list[i])
        cv2.imwrite(output_file, output_frame)

def apply_temporal_smoothing(depth_frames_folder, n=5, kernel_size=3, sigma=1):
    file_list = sorted(os.listdir(depth_frames_folder))
    num_files = len(file_list)

    # Process frames in parallel using multiple CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_idx = {executor.submit(process_frame, idx, file_list, depth_frames_folder, n, kernel_size, sigma): idx for idx in range(num_files)}
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=num_files, desc="Applying TGausiann"):
            idx = future_to_idx[future]
            smoothed_map = future.result()
            output_file = os.path.join(depth_frames_folder, file_list[idx])
            smoothed_map_uint16 = np.uint16(smoothed_map)  # Convert to uint16
            cv2.imwrite(output_file, smoothed_map_uint16)

def process_frame(idx, file_list, depth_frames_folder, n, kernel_size, sigma):
    depth_maps_queue = deque()

    for i in range(max(0, idx - n // 2), min(idx + n // 2 + 1, len(file_list))):
        file = file_list[i]
        if file.endswith(".png"):
            depth_map = cv2.imread(os.path.join(depth_frames_folder, file), cv2.IMREAD_UNCHANGED)
            depth_maps_queue.append(depth_map)

    smoothed_map = smooth_depth_maps(depth_maps_queue, kernel_size, sigma)
    return smoothed_map


def smooth_depth_maps(depth_maps_queue, kernel_size, sigma):
    # Check that all depth maps have the same shape
    first_map_shape = depth_maps_queue[0].shape
    for depth_map in depth_maps_queue:
        if depth_map.shape != first_map_shape:
            raise ValueError("All depth maps must have the same shape.")

    num_maps = len(depth_maps_queue)
    smoothed_map = np.zeros_like(depth_maps_queue[0], dtype=np.float32)

    for i, depth_map in enumerate(depth_maps_queue):
        weight = np.exp(-((num_maps // 2 - i) ** 2) / (2 * sigma ** 2))
        smoothed_map += weight * depth_map

    smoothed_map /= np.sum([np.exp(-((num_maps // 2 - i) ** 2) / (2 * sigma ** 2)) for i in range(num_maps)])

    return smoothed_map.astype(np.uint16)

# Define a function to process each frame
def combine_frame(color_file, depth_file, color_frames_folder, depth_frames_folder, combined_frames_folder):
    color_frame = cv2.imread(os.path.join(color_frames_folder, color_file))
    depth_frame = cv2.imread(os.path.join(depth_frames_folder, depth_file))

    # Resize depth_frame to match color_frame dimensions
    depth_frame_resized = cv2.resize(depth_frame, (color_frame.shape[1], color_frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    combined_frame = np.concatenate((color_frame, depth_frame_resized), axis=1)
    combined_file = os.path.join(combined_frames_folder, color_file)
    cv2.imwrite(combined_file, combined_frame)

def combine_frames_into_video(input_video, color_frames_folder, depth_frames_folder, output_video):
    # Combine the color and depth frames into a single video and save it as output_video

    # Create a folder to store the combined frames
    combined_frames_folder = './out/combined_frames/'
    clear_folder(combined_frames_folder)

    # Iterate through the files in both folders and combine them side by side
    color_files = sorted(os.listdir(color_frames_folder))
    depth_files = sorted(os.listdir(depth_frames_folder))

    # Use multiple processes to combine frames in parallel
    with multiprocessing.Pool() as pool:
        results = []
        for color_file, depth_file in zip(color_files, depth_files):
            result = pool.apply_async(combine_frame, (color_file, depth_file, color_frames_folder, depth_frames_folder, combined_frames_folder))
            results.append(result)
        
        # Wait for all processes to finish
        for result in tqdm(results, total=len(results), desc="Combining frames"):
            result.get()

        ffprobe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "{input_video}"'
        frame_rate = subprocess.check_output(ffprobe_cmd, shell=True, text=True).strip()

        # Use ffmpeg to create a video from the combined frames
        input_pattern = os.path.join(combined_frames_folder, 'frame_%07d.png')
        command = f'ffmpeg -y -r {frame_rate} -i "{input_pattern}" -i "{input_video}" -c:v hevc_nvenc -pix_fmt yuv420p -preset fast -crf 28 -map 0:v -map 1:a -c:a copy -shortest "{output_video}"'
    
    # Run the command and redirect the output to subprocess.PIPE
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the command to finish and get the output and errors
    output, errors = proc.communicate()

    # If there were any errors, print them
    if proc.returncode != 0:
        print(errors.decode())

def generate_depth_video(input_video, output_video, doTAA = True, doTGaussian = False):
    # Split video into individual frames
    main_start_time = time.time()
    start_time = time.time()
    color_frames_folder = split_video_into_frames(input_video)
    # color_frames_folder = "./out/color_frames/"
    elapsed_time = time.time() - start_time
    print(f"Splitting video into frames took {elapsed_time:.2f} seconds.")

    # Generate depth maps from the frames
    start_time = time.time()
    depth_frames_folder = generate_depth_from_frames(color_frames_folder)
    # depth_frames_folder = "./out/depth_frames/"
    elapsed_time = time.time() - start_time
    print(f"Generating depth maps took {elapsed_time:.2f} seconds.")

    # Apply temporal smoothing to the depth maps
    start_time = time.time()

    if doTGaussian:
        apply_temporal_smoothing(depth_frames_folder)

    if doTAA:
        apply_TAA(depth_frames_folder)
        apply_TAA(depth_frames_folder)

    elapsed_time = time.time() - start_time
    print(f"Applying temporal smoothing took {elapsed_time:.2f} seconds.")

    # Combine the color and depth frames into a single video
    start_time = time.time()
    combine_frames_into_video(input_video, color_frames_folder, depth_frames_folder, output_video)
    elapsed_time = time.time() - start_time
    total_elapsed_time = time.time() - main_start_time
    print(f"Combining frames into video took {elapsed_time:.2f} seconds.")
    print(f"Total time processing video took {total_elapsed_time:.2f} seconds.")

def convert_to_mp4(input_video, output_video, duration=None):
    duration_option = f'-t {duration}' if duration else ''
    command = f'ffmpeg -y {duration_option} -i "{input_video}" -c:v libx264 -preset fast -crf 28 -c:a aac "{output_video}"'
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = proc.communicate()

def process_all_videos(video_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_formats = [".mp4", ".mkv", ".mov", ".avi", ".m4v"]

    for video_file in os.listdir(unformatted_video_folder):
        file_extension = os.path.splitext(video_file)[-1].lower()
        if file_extension in supported_formats:
            input_video = os.path.join(unformatted_video_folder, video_file)
            converted_video = os.path.join(video_folder, os.path.splitext(video_file)[0] + ".mp4")
            print(f"Converting video: {input_video} to {converted_video}")
            convert_to_mp4(input_video, converted_video)
            output_video = os.path.join(output_folder, "depth_" + os.path.splitext(video_file)[0] + ".mp4")
            print(f"Processing video: {converted_video}")
            generate_depth_video(converted_video, output_video)


if __name__ == "__main__":
    unformatted_video_folder = "./unformatted_videos/"
    video_folder = "./videos/"
    output_folder = "./depth_videos/"
    process_all_videos(video_folder, output_folder)