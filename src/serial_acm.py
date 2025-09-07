import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
import src.config as config

def _serial_snake_iteration(snake, energy_interpolator, search_window):
    """
    Performs a single iteration of the serial snake algorithm.
    """
    n_points = len(snake)
    new_snake = snake.copy()

    # Calculate the average distance between points for the continuity term
    avg_dist = np.mean(np.linalg.norm(np.roll(snake, -1, axis=0) - snake, axis=1))

    # Iterate over each point in the snake
    for j in range(n_points):
        current_point = snake[j]
        prev_point = snake[(j - 1 + n_points) % n_points]
        next_point = snake[(j + 1) % n_points]

        min_energy = float('inf')
        best_position = current_point

        # Search in the neighborhood of the current point
        for move in search_window:
            candidate_pos = current_point + move

            # Internal Energy
            e_continuity = np.abs(avg_dist - np.linalg.norm(candidate_pos - prev_point))**2
            e_curvature = np.linalg.norm(prev_point - 2 * candidate_pos + next_point)**2
            e_internal = config.ALPHA * e_continuity + config.BETA * e_curvature

            # External (Image) Energy
            e_image = energy_interpolator(candidate_pos[1], candidate_pos[0], grid=False)

            total_energy = e_internal + config.W_EDGE * e_image

            if total_energy < min_energy:
                min_energy = total_energy
                best_position = candidate_pos

        new_snake[j] = best_position

    return new_snake

def run_serial(image, initial_snake):
    """
    Runs the serial active contour model algorithm.
    """
    snake = initial_snake.copy()

    # Pre-computation
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    image_energy = -cv2.magnitude(grad_x, grad_y)
    cv2.normalize(image_energy, image_energy, 0, 1, cv2.NORM_MINMAX)

    h, w = image.shape
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    energy_interpolator = RectBivariateSpline(y_coords, x_coords, image_energy, kx=2, ky=2, s=0)

    search_window = np.array([
        (i, j) for i in range(-config.N_SEARCH, config.N_SEARCH + 1)
        for j in range(-config.N_SEARCH, config.N_SEARCH + 1)
    ])

    # Main iterative loop
    for i in range(config.N_ITERATIONS):
        snake = _serial_snake_iteration(snake, energy_interpolator, search_window)

    return snake
