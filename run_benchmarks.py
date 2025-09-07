import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from utils import preprocess_image, initialize_snake, visualize_and_save
from serial_acm import run_serial
from parallel_acm import run_parallel
import config

def run_single_benchmark(image_path, mode, n_runs=3):
    """
    Runs a single benchmark for a given image and mode, averaging over n_runs.
    Handles warm-up for the parallel mode.
    """
    processed_image, _ = preprocess_image(image_path)
    initial_snake = initialize_snake(processed_image.shape, config.N_POINTS)

    if mode == 'parallel':
        # Warm-up run for Numba JIT compilation
        print("  Warming up Numba JIT...")
        _ = run_parallel(processed_image, initial_snake)
        print("  Warm-up complete.")

    total_time = 0
    for i in range(n_runs):
        start_time = time.time()
        if mode == 'serial':
            run_serial(processed_image, initial_snake)
        else: # parallel
            run_parallel(processed_image, initial_snake)
        end_time = time.time()
        total_time += (end_time - start_time)
        print(f"  Run {i+1}/{n_runs}: {end_time - start_time:.4f}s")

    return total_time / n_runs

def main():
    # --- 1. Run Benchmarks ---
    image_files = ['data/circle_256.png', 'data/circle_512.png', 'data/circle_1024.png']
    results_data = []

    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')

    for image_path in image_files:
        if not os.path.exists(image_path):
            print(f"Warning: Test image {image_path} not found. Skipping.")
            continue

        size = int(os.path.basename(image_path).split('_')[1].split('.')[0])
        print(f"\n--- Benchmarking for image size: {size}x{size} ---")

        # Run serial
        print("Running serial benchmark...")
        serial_time = run_single_benchmark(image_path, 'serial')
        results_data.append({'size': size, 'mode': 'serial', 'time': serial_time})
        print(f"  Average serial time: {serial_time:.4f}s")

        # Run parallel
        print("Running parallel benchmark...")
        parallel_time = run_single_benchmark(image_path, 'parallel')
        results_data.append({'size': size, 'mode': 'parallel', 'time': parallel_time})
        print(f"  Average parallel time: {parallel_time:.4f}s")

    # --- 2. Save results to CSV ---
    log_path = 'results/performance_log.csv'
    try:
        with open(log_path, 'w', newline='') as csvfile:
            fieldnames = ['size', 'mode', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_data)
        print(f"\nPerformance log saved to {log_path}")
    except IOError:
        print(f"Error: Could not write to {log_path}")
        return

    # --- 3. Analyze and Plot ---
    if not results_data:
        print("No results to plot. Exiting.")
        return

    df = pd.DataFrame(results_data)

    # Calculate speedup
    speedup_data = []
    for size in sorted(df['size'].unique()):
        serial_time = df[(df['size'] == size) & (df['mode'] == 'serial')]['time'].iloc[0]
        parallel_time = df[(df['size'] == size) & (df['mode'] == 'parallel')]['time'].iloc[0]
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        speedup_data.append({'size': size, 'speedup': speedup})
        print(f"Speedup for {size}x{size}: {speedup:.2f}x")

    speedup_df = pd.DataFrame(speedup_data)

    # Plot 1: Execution Time vs. Image Size
    plt.figure(figsize=(10, 6))
    for mode in ['serial', 'parallel']:
        subset = df[df['mode'] == mode]
        plt.plot(subset['size'], subset['time'], marker='o', linestyle='-', label=mode)

    plt.title('Execution Time vs. Image Size')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.xticks(df['size'].unique())
    plt.grid(True)
    plt.legend()
    time_plot_path = 'results/execution_time_vs_size.png'
    plt.savefig(time_plot_path)
    print(f"Execution time plot saved to {time_plot_path}")
    plt.close()

    # Plot 2: Speedup Ratio vs. Image Size
    if not speedup_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(speedup_df['size'], speedup_df['speedup'], marker='o', linestyle='-')

        plt.title('Speedup Ratio vs. Image Size')
        plt.xlabel('Image Size (pixels)')
        plt.ylabel('Speedup Ratio (Serial Time / Parallel Time)')
        plt.xticks(speedup_df['size'].unique())
        plt.axhline(y=1, color='r', linestyle='--', label='No Speedup')
        plt.grid(True)
        plt.legend()
        speedup_plot_path = 'results/speedup_vs_size.png'
        plt.savefig(speedup_plot_path)
        print(f"Speedup plot saved to {speedup_plot_path}")
        plt.close()

if __name__ == '__main__':
    main()
