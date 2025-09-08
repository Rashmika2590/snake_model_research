import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numba


def preprocess_image(image_path):
    """
    Loads an image, converts it to grayscale, and applies Gaussian smoothing.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return processed_image, original_image


def initialize_snake(image_shape, n_points):
    """
    Initializes a circular snake in the center of the image.
    """
    height, width = image_shape
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) * 0.4  # 40% of smaller dimension

    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center_x + radius * np.cos(t)
    y = center_y + radius * np.sin(t)
    snake = np.vstack((x, y)).T
    return snake.astype(np.float32)


def visualize_and_save(image, contour, output_path):
    """
    Draws the contour on the image and saves it.
    """
    contour_img = image.copy()
    points = contour.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(contour_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(output_path, contour_img)
    print(f"Result saved to {output_path}")


def get_manual_contour(image):
    """
    Allows the user to draw a contour on the image manually.
    """
    points = []
    window_name = "Draw Contour - Press 'Enter' to finish, 'c' to clear"

    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_circle)

    print("Draw the initial contour by clicking points.")
    print("Press 'Enter' to finalize, 'c' to clear, 'q' to quit.")

    temp_img = image.copy()
    while True:
        img_draw = temp_img.copy()
        if len(points) > 0:
            for point in points:
                cv2.circle(img_draw, tuple(point), 3, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.polylines(img_draw, [np.array(points, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=1)

        cv2.imshow(window_name, img_draw)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # Enter
            if len(points) < 3:
                print("Select at least 3 points.")
                continue
            break
        elif key == ord('c'):  # Clear
            points = []
            print("Points cleared. Start over.")
        elif key == ord('q'):  # Quit
            points = []
            break

    cv2.destroyWindow(window_name)
    if not points:
        return None
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

    colormap = plt.colormaps.get_cmap('hsv')
    colors = [colormap(i) for i in np.linspace(0, 1, n_threads)]

    for i, chunk in enumerate(chunks):
        ax.plot(chunk[:, 0], chunk[:, 1], color=colors[i], linewidth=3, label=f'Thread {i}')

    ax.set_title(f'Parallel Workload Division ({n_threads} Threads)')
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

    points = snake.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    unique_threads = np.unique(thread_map)
    n_threads = len(unique_threads)
    thread_to_color_idx = {tid: i for i, tid in enumerate(unique_threads)}

    colormap = plt.colormaps.get_cmap('hsv')
    colors = [colormap(i) for i in np.linspace(0, 1, n_threads)]
    segment_colors = [colors[thread_to_color_idx[thread_map[i]]] for i in range(len(snake) - 1)]

    closing_segment = np.array([[snake[-1], snake[0]]])
    segments = np.vstack([segments, closing_segment])
    segment_colors.append(colors[thread_to_color_idx[thread_map[-1]]])

    lc = LineCollection(segments, colors=segment_colors, linewidths=3)
    ax.add_collection(lc)

    plot_snake = np.vstack([snake, snake[0]])
    ax.plot(plot_snake[:, 0], plot_snake[:, 1], '--k', linewidth=1, label='Final Contour')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[thread_to_color_idx[tid]], lw=4, label=f'Thread {tid}') for tid in unique_threads]
    legend_elements.append(Line2D([0], [0], color='black', linestyle='--', lw=1, label='Final Contour'))
    ax.legend(handles=legend_elements)

    ax.set_xticks([]), ax.set_yticks([])
    plt.savefig(output_path)
    print(f"Parallel results visualization saved to {output_path}")
    plt.close(fig)
