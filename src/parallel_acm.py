import numpy as np
import cv2
import numba
import src.config as config

@numba.njit
def bilinear_interpolate(image, x, y):
    """
    Performs bilinear interpolation on an image at a given (x, y) point.
    This is a Numba-compatible replacement for scipy's interpolators.
    """
    h, w = image.shape
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Return 0 for points outside the image
    if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h:
        return 0.0

    # Get the values of the four corners
    q11 = image[y0, x0]
    q21 = image[y0, x1]
    q12 = image[y1, x0]
    q22 = image[y1, x1]

    # Calculate interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    # Return interpolated value
    return wa * q11 + wb * q21 + wc * q12 + wd * q22


@numba.njit(parallel=True)
def _parallel_snake_iteration(snake, image_energy, search_window, alpha, beta, w_edge):
    """
    A single iteration of the snake algorithm, parallelized with Numba.
    """
    n_points = len(snake)
    new_snake = np.zeros_like(snake)

    # Manually roll the snake array
    rolled_snake = np.empty_like(snake)
    rolled_snake[:-1] = snake[1:]
    rolled_snake[-1] = snake[0]

    # Compute distances for continuity energy
    diffs = rolled_snake - snake
    norms = np.sqrt(np.sum(diffs**2, axis=1))
    avg_dist = np.mean(norms)

    for j in numba.prange(n_points):
        current_point = snake[j]
        prev_point = snake[(j - 1 + n_points) % n_points]
        next_point = snake[(j + 1) % n_points]

        min_energy = np.inf
        best_position = current_point

        for move in search_window:
            candidate_pos = current_point + move

            # Internal energy
            e_continuity = np.abs(avg_dist - np.linalg.norm(candidate_pos - prev_point))**2
            e_curvature = np.linalg.norm(prev_point - 2 * candidate_pos + next_point)**2
            e_internal = alpha * e_continuity + beta * e_curvature

            # External energy (interpolated)
            e_image = bilinear_interpolate(image_energy, candidate_pos[0], candidate_pos[1])
            total_energy = e_internal + w_edge * e_image

            if total_energy < min_energy:
                min_energy = total_energy
                best_position = candidate_pos

        new_snake[j] = best_position

    return new_snake


def run_parallel(image, initial_snake):
    """
    Runs the parallel active contour model algorithm.
    Returns the final snake.
    """
    snake = initial_snake.copy()

    # Compute image energy
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    image_energy = -cv2.magnitude(grad_x, grad_y)
    cv2.normalize(image_energy, image_energy, 0, 1, cv2.NORM_MINMAX)

    # Define search window
    search_window = np.array([
        (i, j) for i in range(-config.N_SEARCH, config.N_SEARCH + 1)
        for j in range(-config.N_SEARCH, config.N_SEARCH + 1)
    ], dtype=np.float32)

    # Iteratively update snake
    for i in range(config.N_ITERATIONS):
        snake = _parallel_snake_iteration(
            snake,
            image_energy,
            search_window,
            config.ALPHA,
            config.BETA,
            config.W_EDGE
        )

    return snake
