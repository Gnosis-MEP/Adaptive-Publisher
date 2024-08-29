import time
import zlib
# pip install lz4
# pip install python-snappy
import snappy
import lz4.frame
import numpy as np
from PIL import Image

def compress_zlib(nd_array_bytes, level):
    compressed_data = zlib.compress(nd_array_bytes, level=level)
    return compressed_data

def compress_lz4(nd_array_bytes, level):
    compressed_data = lz4.frame.compress(nd_array_bytes)
    return compressed_data

def compress_snappy(nd_array_bytes, level):
    compressed_data = snappy.compress(nd_array_bytes)
    return compressed_data

def measure_compression_time(image_path, method,compression_level, iterations=100):
    times = []

    for _ in range(iterations):
        # Load the image from file
        image = Image.open(image_path)
        img_numpy_array = np.array(image)

        # Convert numpy array to bytes
        nd_array_bytes = img_numpy_array.tobytes(order='C')

        # Measure compression time
        start_time = time.perf_counter()
        compressed_data = method(nd_array_bytes, level=compression_level)
        end_time = time.perf_counter()

        # Calculate and store the time taken
        compression_time = end_time - start_time
        times.append(compression_time)

        # Delete the image variable to free memory
        del image
        del img_numpy_array
        del nd_array_bytes
        del compressed_data
        print('_')

    # Calculate average time
    average_time = sum(times) / len(times)
    return average_time

# Path to your image file
image_path = 'frame_1.png'  # Replace with the actual path to your image

# Compression level (1 for fast, 9 for maximum compression)
compression_level = 1  # You can change this to any level from 1 to 9

# Measure average compression time
average_compression_time = measure_compression_time(image_path, compress_zlib, compression_level)
print(f'(ZLIB) Average compression time over 100 iterations: {average_compression_time:.6f} seconds')

average_compression_time = measure_compression_time(image_path, compress_lz4, compression_level)
print(f'(LZ4) Average compression time over 100 iterations: {average_compression_time:.6f} seconds')

average_compression_time = measure_compression_time(image_path, compress_snappy, compression_level)
print(f'(SNAPPY) Average compression time over 100 iterations: {average_compression_time:.6f} seconds')

