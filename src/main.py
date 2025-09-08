import argparse
import time
import os
import sys
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

# Add src to path to allow imports when running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
from src.utils import (
    preprocess_image, initialize_snake, visualize_and_save, get_manual_contour,
    visualize_parallel_setup, visualize_parallel_results
)
from src.serial_acm import run_serial, _serial_snake_iteration
from src.parallel_acm import run_parallel, _parallel_snake_iteration

def main():
    parser = argparse.ArgumentParser(description="Active Contour Model (Snake) for Image Segmentation")

    # Execution modes
    parser.add_argument('--image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--mode', type=str, choices=['serial', 'parallel'], default='serial', help="Execution mode for batch processing.")

    # Initialization modes
    parser.add_argument('--init', type=str, choices=['auto', 'manual'], default='auto', help="Contour initialization method.")

    # Visualization modes
    parser.add_argument('--realtime', action='store_true', help="Enable real-time OpenCV visualization.")
    parser.add_argument('--vis-serial', action='store_true', help="Enable Matplotlib serial visualization.")
    parser.add_argument('--vis-parallel-setup', action='store_true', help="Plot parallel workload division.")
    parser.add_argument('--vis-parallel-results', action='store_true', help="Plot parallel computation results.")

    # Output
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save output images.")

    args = parser.parse_args()

    # --- 1. Load and preprocess image ---
    print(f"Loading image from {args.image}...")
    processed_image, original_image = preprocess_image(args.image)

    # --- 2. Initialize snake ---
    if args.init == 'manual':
        print("Manual initialization selected. Please draw on the image.")
        initial_snake = get_manual_contour(original_image)
        if initial_snake is None:
            print("Initialization cancelled. Exiting.")
            sys.exit()
    else:
        print("Automatic initialization selected...")
        initial_snake = initialize_snake(processed_image.shape, config.N_POINTS)

    snake = initial_snake.copy()
    final_snake = snake
    thread_map = None

    # --- Optional: Visualize Parallel Setup ---
    if args.vis_parallel_setup and args.mode == 'parallel':
        setup_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image))[0]}_parallel_setup.png")
        visualize_parallel_setup(original_image, snake, setup_path)

    # --- 3. Run ACM ---
    if args.vis_serial:
        # Matplotlib serial visualization
        print("Running Matplotlib real-time serial visualization...")
        h, w = processed_image.shape
        x_coords, y_coords = np.arange(w), np.arange(h)
        grad_x = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=5)
        image_energy = -cv2.magnitude(grad_x, grad_y)
        cv2.normalize(image_energy, image_energy, 0, 1, cv2.NORM_MINMAX)
        energy_interpolator = RectBivariateSpline(y_coords, x_coords, image_energy, kx=2, ky=2, s=0)
        search_window = np.array([(i, j) for i in range(-config.N_SEARCH, config.N_SEARCH+1)
                                  for j in range(-config.N_SEARCH, config.N_SEARCH+1)])
        plt.ion()
        fig, ax = plt.subplots()
        for i in range(config.N_ITERATIONS):
            snake = _serial_snake_iteration(snake, energy_interpolator, search_window)
            if not plt.fignum_exists(fig.number):
                print("Plot window closed. Stopping evolution.")
                break
            ax.clear()
            ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plot_snake = np.vstack([snake, snake[0]])
            ax.plot(plot_snake[:, 0], plot_snake[:, 1], '-r', linewidth=2)
            ax.set_title(f"Serial Evolution - Iteration: {i+1}")
            ax.set_xticks([]), ax.set_yticks([])
            plt.pause(0.01)
        plt.ioff()
        final_snake = snake
        plt.show()

    elif args.realtime:
        # OpenCV real-time visualization
        print("Running OpenCV real-time visualization. Press 'q' to stop.")
        grad_x = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=5)
        image_energy = -cv2.magnitude(grad_x, grad_y)
        cv2.normalize(image_energy, image_energy, 0, 1, cv2.NORM_MINMAX)
        h, w = processed_image.shape
        x_coords, y_coords = np.arange(w), np.arange(h)
        energy_interpolator = RectBivariateSpline(y_coords, x_coords, image_energy, kx=2, ky=2, s=0)
        search_window = np.array([(i, j) for i in range(-config.N_SEARCH, config.N_SEARCH+1)
                                  for j in range(-config.N_SEARCH, config.N_SEARCH+1)], dtype=np.float32)
        window_name = "Active Contour Evolution (OpenCV)"
        cv2.namedWindow(window_name)
        for i in range(config.N_ITERATIONS):
            if args.mode == 'serial':
                snake = _serial_snake_iteration(snake, energy_interpolator, search_window)
            else:
                if i == 0:
                    _ = _parallel_snake_iteration(snake, image_energy, search_window,
                                                  config.ALPHA, config.BETA, config.W_EDGE)
                snake, _ = _parallel_snake_iteration(snake, image_energy, search_window,
                                                     config.ALPHA, config.BETA, config.W_EDGE)
            vis_image = original_image.copy()
            points = snake.astype(np.int32).reshape((-1,1,2))
            cv2.polylines(vis_image, [points], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(vis_image, f"Iteration: {i+1}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow(window_name, vis_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        final_snake = snake

    else:
        # Standard batch execution (no visualization)
        start_time = time.time()
        if args.mode == 'serial':
            print("Running Serial ACM...")
            final_snake = run_serial(processed_image, snake)
        else:
            print("Running Parallel ACM...")
            final_snake, thread_map = run_parallel(processed_image, snake)
        end_time = time.time()
        print(f"Execution time ({args.mode}): {end_time - start_time:.4f} seconds")

    # --- Optional: Visualize Parallel Results ---
    if args.vis_parallel_results and args.mode == 'parallel' and thread_map is not None:
        results_path = os.path.join(args.output_dir,
                                    f"{os.path.splitext(os.path.basename(args.image))[0]}_parallel_results.png")
        visualize_parallel_results(original_image, final_snake, thread_map, results_path)

    # --- 4. Save Final Result ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    base_filename = os.path.splitext(os.path.basename(args.image))[0]
    output_filename = f"{base_filename}_{args.mode}_result.png"
    output_path = os.path.join(args.output_dir, output_filename)
    visualize_and_save(original_image, final_snake, output_path)

if __name__ == '__main__':
    main()
