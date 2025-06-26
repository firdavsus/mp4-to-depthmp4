
import subprocess
import os
ffprobe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 ./videos/'
frame_rate = subprocess.check_output(ffprobe_cmd, shell=True, text=True).strip()

# Use ffmpeg to create a video from the combined frames
input_pattern = os.path.join("./out/combined_frames/", 'frame_%07d.png')
command = f'ffmpeg -y -r {frame_rate} -i "{input_pattern}" -i ./videos/ -c:v hevc_nvenc -pix_fmt yuv420p -preset fast -crf 28 -map 0:v -map 1:a -c:a copy -shortest loli.mp4'