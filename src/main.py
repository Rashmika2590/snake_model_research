import argparse
import time
import os
import sys
 feat/active-contour-model-project
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline

 main

# Add src to path to allow imports when running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
 feat/active-contour-model-project
from src.utils import preprocess_image, initialize_snake, visualize_and_save, get_manual_contour
from src.serial_acm import run_serial, _serial_snake_iteration
from src.parallel_acm import run_parallel, _parallel_snake_iteration

from src.utils import preprocess_image, initialize_snake, visualize_and_save
from src.serial_acm import run_serial
from src.parallel_acm import run_parallel
 main

def main():
    parser = argparse.ArgumentParser(description="Active Contour Model (Snake) for Image Segmentation")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--mode', type=str, choices=['serial', 'parallel'], default='serial', help="Execution mode.")
 feat/active-contour-model-project
    parser.add_argument('--init', type=str, choices=['auto', 'manual'], default='auto', help="Contour initialization method.")
    parser.add_argument('--realtime', action='store_true', help="Enable real-time visualization of the snake evolution. NOTE: Does not work in headless environments.")

 main
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save the output images.")
    args = parser.parse_args()

    # --- 1. Load and Preprocess Image ---
    print(f"Loading image from {args.image}...")
    processed_image, original_image = preprocess_image(args.image)

    # --- 2. Initialize Snake ---
 feat/active-contour-model-project
    if args.init == 'manual':
        print("Manual initialization selected. Please draw on the image.")
        # NOTE: This will fail in a headless environment.
        initial_snake = get_manual_contour(original_image)
        if initial_snake is None:
            print("Initialization cancelled. Exiting.")
            return
    else:
        print("Initializing snake automatically...")
        initial_snake = initialize_snake(processed_image.shape, config.N_POINTS)

    snake = initial_snake.copy()

    # --- 3. Run Active Contour Model ---
    if not args.realtime:
        # Standard, non-realtime execution
        start_time = time.time()
        if args.mode == 'serial':
            print("Running Serial Active Contour Model...")
            final_snake = run_serial(processed_image, snake)
        else: # parallel
            print("Running Parallel Active Contour Model...")
            final_snake = run_parallel(processed_image, snake)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time ({args.mode}): {execution_time:.4f} seconds")

    else:
        # Real-time visualization mode
        print("Running in real-time visualization mode. Press 'q' to stop.")
        # NOTE: This will fail in a headless environment.

        # Pre-computation needed for the iteration functions
        grad_x = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=5)
        image_energy = -cv2.magnitude(grad_x, grad_y)
        cv2.normalize(image_energy, image_energy, 0, 1, cv2.NORM_MINMAX)

        search_window = np.array([
            (i, j) for i in range(-config.N_SEARCH, config.N_SEARCH + 1)
            for j in range(-config.N_SEARCH, config.N_SEARCH + 1)
        ], dtype=np.float32)

        # Specific pre-computation for serial mode's interpolator
        h, w = processed_image.shape
        x_coords = np.arange(w)
        y_coords = np.arange(h)
        energy_interpolator = RectBivariateSpline(y_coords, x_coords, image_energy, kx=2, ky=2, s=0)

        window_name = "Active Contour Evolution"
        cv2.namedWindow(window_name)

        for i in range(config.N_ITERATIONS):
            # Call the appropriate single-iteration function
            if args.mode == 'serial':
                snake = _serial_snake_iteration(snake, energy_interpolator, search_window)
            else: # parallel
                # Warm-up Numba on the first iteration
                if i == 0:
                    _ = _parallel_snake_iteration(snake, image_energy, search_window, config.ALPHA, config.BETA, config.W_EDGE)
                snake = _parallel_snake_iteration(snake, image_energy, search_window, config.ALPHA, config.BETA, config.W_EDGE)

            # Visualization
            vis_image = original_image.copy()
            points = snake.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(vis_image, f"Iteration: {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(window_name, vis_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        final_snake = snake

    # --- 4. Visualize and Save Final Result ---
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
 main
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    base_filename = os.path.splitext(os.path.basename(args.image))[0]
    output_filename = f"{base_filename}_{args.mode}_result.png"
    output_path = os.path.join(args.output_dir, output_filename)

    visualize_and_save(original_image, final_snake, output_path)

if __name__ == '__main__':
    main()
