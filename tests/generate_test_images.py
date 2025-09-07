import os
import cv2
import numpy as np

def create_image_with_circle(width, height, radius_ratio, filename):
    """
    Creates a black image with a white circle in the center.

    Args:
        width (int): Width of the image.
        height (int): Height of the image.
        radius_ratio (float): The radius of the circle as a ratio of the smaller dimension.
        filename (str): The path to save the image.
    """
    # Create a black image
    img = np.zeros((height, width), dtype=np.uint8)

    # Get the center and radius
    center_x, center_y = width // 2, height // 2
    radius = int(min(width, height) * radius_ratio)

    # Draw a white circle
    cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), -1)

    # Save the image
    cv2.imwrite(filename, img)
    print(f"Created {filename}")

if __name__ == "__main__":
    # This script assumes it is run from the project root directory
    data_dir = "data"

    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Generate images of different sizes
    create_image_with_circle(256, 256, 0.3, os.path.join(data_dir, "circle_256.png"))
    create_image_with_circle(512, 512, 0.3, os.path.join(data_dir, "circle_512.png"))
    create_image_with_circle(1024, 1024, 0.3, os.path.join(data_dir, "circle_1024.png"))
