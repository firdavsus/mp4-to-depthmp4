import os
import re
import cv2
import numpy as np
import subprocess

def extract_frame_index(filename):
    match = re.search(r'frame_(\d+)', filename)
    return int(match.group(1)) if match else None

def create_rgba_frames(rgb_folder, depth_folder, rgba_folder):
    os.makedirs(rgba_folder, exist_ok=True)

    rgb_files = sorted(
        [f for f in os.listdir(rgb_folder) if f.startswith("frame_") and f.endswith(".png")],
        key=extract_frame_index
    )
    depth_files = {
        extract_frame_index(f): f for f in os.listdir(depth_folder)
        if f.endswith(".png") and "-dpt_beit_large_512" in f
    }

    for i, rgb_file in enumerate(rgb_files):
        frame_idx = extract_frame_index(rgb_file)
        if frame_idx not in depth_files:
            print(f"⚠️ Depth file not found for {rgb_file}, skipping.")
            continue

        rgb_path = os.path.join(rgb_folder, rgb_file)
        depth_path = os.path.join(depth_folder, depth_files[frame_idx])
        rgba_path = os.path.join(rgba_folder, f"frame_{i:07d}.png")

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        if rgb is None or depth is None:
            print(f"⚠️ Skipping {rgb_file} — RGB or depth image not found.")
            continue

        if rgb.shape[:2] != depth.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        alpha = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX).astype(np.uint8)
        rgba = cv2.merge((rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], alpha))
        cv2.imwrite(rgba_path, rgba)

    print("✅ RGBA frames created with depth as alpha channel.")

def extract_frame_rate(reference_video):
    ffprobe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        reference_video
    ]
    return subprocess.check_output(ffprobe_cmd, text=True).strip()

def create_video_with_audio(rgba_folder, output_path, reference_video, format='mov'):
    frame_rate = extract_frame_rate(reference_video)
    input_pattern = os.path.join(rgba_folder, "frame_%07d.png")

    if format == 'mov':
        output_path = output_path.replace('.mp4', '.mov')
        command = [
            "ffmpeg", "-y",
            "-framerate", frame_rate,
            "-i", input_pattern,
            "-i", reference_video,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "qtrle",  # QuickTime RLE for alpha
            "-pix_fmt", "argb",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
    elif format == 'webm':
        output_path = output_path.replace('.mp4', '.webm')
        command = [
            "ffmpeg", "-y",
            "-framerate", frame_rate,
            "-i", input_pattern,
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0",
            output_path
        ]
    else:
        raise ValueError("Unsupported format. Use 'mov' or 'webm'.")

    subprocess.run(command)
    print(f"✅ Final Holobox video created at: {output_path}")


# === USAGE CONFIG ===
rgb_folder = "./out/color_frames"
depth_folder = "./out/depth_frames"
rgba_folder = "./out/rgba"
reference_video = "./unformatted_videos/aaa.mp4"
output_video = "./videos/holobox_ready_video.mov"

# === RUN PIPELINE ===
create_rgba_frames(rgb_folder, depth_folder, rgba_folder)
create_video_with_audio(rgba_folder, output_video, reference_video, format='mov')  # Or 'webm'

