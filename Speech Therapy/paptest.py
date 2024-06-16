import numpy as np
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


def print_gpu_info():
    device = cp.cuda.Device(0)
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    print(f"Using GPU: {properties['name']}")


def process_video(video_path):
    reader = imageio.get_reader(video_path, "ffmpeg")
    frame_count = reader.count_frames()

    # Print GPU info
    print_gpu_info()

    # CPU processing
    start_time = time.time()
    for frame in reader:
        frame = np.array(frame)
        processed_frame = process_frame_cpu(frame)
    cpu_time = time.time() - start_time
    reader.close()

    # GPU processing
    reader = imageio.get_reader(video_path, "ffmpeg")
    cp.cuda.Device(0).use()  # Ensure using the first GPU
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
