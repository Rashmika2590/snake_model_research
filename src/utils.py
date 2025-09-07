import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale, and applies Gaussian smoothing.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The preprocessed grayscale image.
        numpy.ndarray: The original color image for visualization.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image and reduce noise
    processed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    return processed_image, original_image


def initialize_snake(image_shape, n_points):
    """
    Initializes a circular snake in the center of the image.

    Args:
        image_shape (tuple): The shape of the image (height, width).
        n_points (int): The number of points for the snake.

    Returns:
        numpy.ndarray: An array of shape (n_points, 2) representing the snake points.
    """
    height, width = image_shape
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) * 0.4  # 40% of the smaller dimension

    # Generate points for the circle
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center_x + radius * np.cos(t)
    y = center_y + radius * np.sin(t)

    # Reshape to (n_points, 2)
    snake = np.vstack((x, y)).T
    return snake.astype(np.float32)


def visualize_and_save(image, contour, output_path):
    """
    Draws the contour on the image and saves it.

    Args:
        image (numpy.ndarray): The original image.
        contour (numpy.ndarray): The snake contour points.
        output_path (str): The path to save the visualized image.
    """
    # Draw the contour on the image
    contour_img = image.copy()

    # Reshape contour for polylines
    points = contour.astype(np.int32).reshape((-1, 1, 2))

    # Draw the polyline
    cv2.polylines(contour_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Save the image
    cv2.imwrite(output_path, contour_img)
    print(f"Result saved to {output_path}")
