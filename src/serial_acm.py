import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
import src.config as config

def run_serial(image, initial_snake):
    """
    Runs the serial active contour model algorithm using a greedy approach.

    Args:
        image (numpy.ndarray): The preprocessed grayscale image.
        initial_snake (numpy.ndarray): The initial snake contour, shape (N, 2).

    Returns:
        numpy.ndarray: The final snake contour.
    """
    snake = initial_snake.copy()

    # --- Pre-computation ---
    # 1. Calculate the image gradient (for image energy)
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # The image energy is the negative of the gradient magnitude.
    # We want to minimize energy, so we move towards strong gradients (edges).
    image_energy = -cv2.magnitude(grad_x, grad_y)

    # Normalize the image energy to be between 0 and 1 for stability
    cv2.normalize(image_energy, image_energy, 0, 1, cv2.NORM_MINMAX)

    # 2. Create an interpolator for the image energy.
    # This allows us to get energy values at sub-pixel locations.
    h, w = image.shape
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    energy_interpolator = RectBivariateSpline(y_coords, x_coords, image_energy, kx=2, ky=2, s=0)

    # 3. Define the search window for the greedy algorithm
    search_window = np.array([
        (i, j) for i in range(-config.N_SEARCH, config.N_SEARCH + 1)
        for j in range(-config.N_SEARCH, config.N_SEARCH + 1)
    ])

    # --- Main Iterative Loop ---
    for i in range(config.N_ITERATIONS):
        # Calculate the average distance between points for the continuity term
        # This helps maintain the snake's shape and prevents points from clustering.
        avg_dist = np.mean(np.linalg.norm(np.roll(snake, -1, axis=0) - snake, axis=1))

        new_snake = snake.copy()

        # Iterate over each point in the snake
        for j in range(len(snake)):
            current_point = snake[j]

            # Get previous and next points for curvature calculation
            prev_point = snake[(j - 1 + len(snake)) % len(snake)]
            next_point = snake[(j + 1) % len(snake)]

            min_energy = float('inf')
            best_position = current_point

            # Search in the neighborhood of the current point
            for move in search_window:
                candidate_pos = current_point + move

                # --- Calculate Internal Energy ---
                # 1. Continuity Energy: Encourages even spacing of points.
                e_continuity = np.abs(avg_dist - np.linalg.norm(candidate_pos - prev_point))**2

                # 2. Curvature Energy: Penalizes sharp bends.
                e_curvature = np.linalg.norm(prev_point - 2 * candidate_pos + next_point)**2

                e_internal = config.ALPHA * e_continuity + config.BETA * e_curvature

                # --- Calculate External (Image) Energy ---
                # Get the energy at the candidate position using interpolation.
                # The interpolator expects (y, x) coordinates.
                e_image = energy_interpolator(candidate_pos[1], candidate_pos[0], grid=False)

                # --- Total Energy ---
                total_energy = e_internal + config.W_EDGE * e_image

                if total_energy < min_energy:
                    min_energy = total_energy
                    best_position = candidate_pos

            # Update the point in the new snake array
            new_snake[j] = best_position

        # Update the snake for the next iteration
        snake = new_snake

        # Optional: Add a convergence check here later if needed

    return snake
