# import cv2
# import time
# import numpy as np


# def process_frame_cpu(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     return gray


# def process_frame_gpu(frame):
#     gpu_frame = cv2.cuda_GpuMat()
#     gpu_frame.upload(frame)

#     gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
#     gray = gpu_gray.download()
#     return gray


# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Cannot open video file.")
#         return

#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     start_time = time.time()
#     for _ in range(frame_count):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         processed_frame = process_frame_cpu(frame)
#     cpu_time = time.time() - start_time
#     cap.release()

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Cannot open video file.")
#         return

#     # GPU processing
#     start_time = time.time()
#     for _ in range(frame_count):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         processed_frame = process_frame_gpu(frame)
#     gpu_time = time.time() - start_time
#     cap.release()

#     # Output the performance results
#     print(f"CPU processing time: {cpu_time:.2f} seconds")
#     print(f"GPU processing time: {gpu_time:.2f} seconds")


# if __name__ == "__main__":
#     video_path = r"test.mp4"
#     process_video(video_path)


import numpy as np
from PIL import Image
import time
import cupy as cp
import imageio


def process_frame_cpu(frame):
    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
    return gray


def process_frame_gpu(frame):
    gpu_frame = cp.asarray(frame)
    weights = cp.array([0.2989, 0.5870, 0.1140])
    gpu_gray = cp.tensordot(gpu_frame[..., :3], weights, axes=([2], [0]))
    gray = cp.asnumpy(gpu_gray)
    return gray


def process_video(video_path):
    reader = imageio.get_reader(video_path, "ffmpeg")
    frame_count = reader.count_frames()

    start_time = time.time()
    for frame in reader:
        frame = np.array(frame)
        processed_frame = process_frame_cpu(frame)
    cpu_time = time.time() - start_time
    reader.close()

    reader = imageio.get_reader(video_path, "ffmpeg")

    start_time = time.time()
    for frame in reader:
        frame = np.array(frame)
        processed_frame = process_frame_gpu(frame)
    gpu_time = time.time() - start_time
    reader.close()

    # Output the performance results
    print(f"CPU processing time: {cpu_time:.2f} seconds")
    print(f"GPU processing time: {gpu_time:.2f} seconds")


if __name__ == "__main__":
    video_path = r"test.mp4"
    process_video(video_path)
