import cv2
import numpy as np
import numba
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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


def get_manual_contour(image):
    """
    Allows the user to draw a contour on the image manually.

    Args:
        image (numpy.ndarray): The image to draw on.

    Returns:
        numpy.ndarray: An array of shape (n_points, 2) representing the user-drawn snake.
    """
    points = []
    window_name = "Draw Contour - Press 'Enter' to finish, 'c' to clear"

    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_circle)

    print("Please draw the initial contour by clicking points on the image.")
    print("Press 'Enter' to finalize the contour.")
    print("Press 'c' to clear all points and start over.")
    print("Press 'q' to quit.")

    temp_img = image.copy()

    while True:
        # Create a fresh copy for drawing
        img_draw = temp_img.copy()

        # Draw points and lines
        if len(points) > 0:
            for point in points:
                cv2.circle(img_draw, tuple(point), 3, (0, 0, 255), -1)

            if len(points) > 1:
                cv2.polylines(img_draw, [np.array(points, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=1)

        cv2.imshow(window_name, img_draw)

        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # Enter key
            if len(points) < 3:
                print("Warning: Please select at least 3 points.")
                continue
            break
        elif key == ord('c'): # Clear points
            points = []
            print("Points cleared. Please start over.")
        elif key == ord('q'): # Quit
            points = []
            break

    cv2.destroyWindow(window_name)

    if not points:
        return None

    # Convert to numpy array and return
    return np.array(points, dtype=np.float32)


def visualize_parallel_setup(image, snake, output_path):
    """
    Visualizes how the contour is conceptually divided among parallel threads.
    """
    n_threads = numba.get_num_threads()
    if n_threads <= 1:
        print("Not a parallel environment, skipping setup visualization.")
        return

    chunks = np.array_split(snake, n_threads)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Use a modern way to get a colormap
    colormap = plt.colormaps.get_cmap('hsv')
    colors = [colormap(i) for i in np.linspace(0, 1, n_threads)]

    for i, chunk in enumerate(chunks):
        # To make the plot continuous, we plot each chunk as a separate line
        ax.plot(chunk[:, 0], chunk[:, 1], color=colors[i], linewidth=3, label=f'Chunk for Thread {i}')

    ax.set_title(f'Conceptual Parallel Workload Division ({n_threads} Threads)')
    ax.legend()
    ax.set_xticks([]), ax.set_yticks([])
    plt.savefig(output_path)
    print(f"Parallel setup visualization saved to {output_path}")
    plt.close(fig)


def visualize_parallel_results(image, snake, thread_map, output_path):
    """
    Visualizes the final contour, colored by the thread that computed each point.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Create line segments from the snake points
    points = snake.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    unique_threads = np.unique(thread_map)
    n_threads = len(unique_threads)

    # Create a mapping from thread_id to a color index
    thread_to_color_idx = {thread_id: i for i, thread_id in enumerate(unique_threads)}

    colormap = plt.colormaps.get_cmap('hsv')
    colors = [colormap(i) for i in np.linspace(0, 1, n_threads)]

    # Color each segment by the thread ID of its starting point
    segment_colors = [colors[thread_to_color_idx[thread_map[i]]] for i in range(len(snake) - 1)]

    # Add the closing segment
    closing_segment = np.array([[snake[-1], snake[0]]])
    segments = np.vstack([segments, closing_segment])
    segment_colors.append(colors[thread_to_color_idx[thread_map[-1]]])

    lc = LineCollection(segments, colors=segment_colors, linewidths=3, label='Computed by Thread ID')
    ax.add_collection(lc)

    # Plot the final merged contour over the top for clarity
    plot_snake = np.vstack([snake, snake[0]])
    ax.plot(plot_snake[:, 0], plot_snake[:, 1], '--k', linewidth=1, label='Final Merged Contour')

    ax.set_title('Parallel Computation Results by Thread')
    # Create a dummy legend for the colors
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[thread_to_color_idx[tid]], lw=4, label=f'Thread {tid}') for tid in unique_threads]
    legend_elements.append(Line2D([0], [0], color='black', linestyle='--', lw=1, label='Final Merged Contour'))
    ax.legend(handles=legend_elements)

    ax.set_xticks([]), ax.set_yticks([])
    plt.savefig(output_path)
    print(f"Parallel results visualization saved to {output_path}")
    plt.close(fig)
