import argparse
import time
import os
import sys

# Add src to path to allow imports when running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
from src.utils import preprocess_image, initialize_snake, visualize_and_save
from src.serial_acm import run_serial
from src.parallel_acm import run_parallel

def main():
    parser = argparse.ArgumentParser(description="Active Contour Model (Snake) for Image Segmentation")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--mode', type=str, choices=['serial', 'parallel'], default='serial', help="Execution mode.")
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save the output images.")
    args = parser.parse_args()

    # --- 1. Load and Preprocess Image ---
    print(f"Loading image from {args.image}...")
    processed_image, original_image = preprocess_image(args.image)

    # --- 2. Initialize Snake ---
    print("Initializing snake...")
    initial_snake = initialize_snake(processed_image.shape, config.N_POINTS)

    # --- 3. Run Active Contour Model ---
    final_snake = None
    start_time = time.time()

    if args.mode == 'serial':
        print("Running Serial Active Contour Model...")
        final_snake = run_serial(processed_image, initial_snake)
    elif args.mode == 'parallel':
        print("Running Parallel Active Contour Model...")
        final_snake = run_parallel(processed_image, initial_snake)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time ({args.mode}): {execution_time:.4f} seconds")

    # --- 4. Visualize and Save Result ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    base_filename = os.path.splitext(os.path.basename(args.image))[0]
    output_filename = f"{base_filename}_{args.mode}_result.png"
    output_path = os.path.join(args.output_dir, output_filename)

    visualize_and_save(original_image, final_snake, output_path)

if __name__ == '__main__':
    main()
