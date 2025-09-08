# Active Contour Model (Snake) for Image Segmentation

This project implements and evaluates an Active Contour Model (ACM) for image segmentation. It provides a serial implementation, a highly optimized parallel version using Numba, and multiple options for visualization.

## Project Structure

- `/src`: Source code for the ACM implementations, utilities, and configuration.
- `/tests`: Test scripts for ensuring code correctness.
- `/data`: Sample images for testing and benchmarking.
- `/results`: Output images, performance logs, and plots.
- `run_benchmarks.py`: A script to automate the performance evaluation.

## Setup

1.  **Clone the repository.**
2.  **Install dependencies** (preferably in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
3.  **Generate Test Data:**
    ```bash
    python3 tests/generate_test_images.py
    ```

## Usage

### Batch Processing

To run the model non-interactively and save the final result:
```bash
# Run parallel mode (recommended)
python3 src/main.py --image data/circle_256.png --mode parallel
```
The final segmented image will be saved in the `/results` directory.

### Visualization Options

The script provides several flags for visualizing the process.

**Note:** All visualization features require a graphical display (a desktop environment) and will **not** work in a headless environment.

**1. Real-time Serial Evolution (`--vis-serial`)**

Uses `matplotlib` to show the serial snake's evolution iteration by iteration in real-time.

```bash
python3 src/main.py --image <path_to_your_image> --vis-serial
```

**2. Parallel Process Visualization**

These flags provide insight into the parallel execution process. They only work when `--mode parallel` is selected.

*   `--vis-parallel-setup`: Before computation, saves a plot showing how the snake contour is conceptually divided among parallel threads.
*   `--vis-parallel-results`: After computation, saves a plot of the final contour, with segments colored according to the thread ID that computed them.

```bash
# Run the parallel mode and generate both visualization plots
python3 src/main.py --image data/circle_256.png --mode parallel --vis-parallel-setup --vis-parallel-results
```
The generated plots will be saved to the `/results` directory.

**3. Manual Contour Initialization (`--init manual`)**

Draw the initial contour yourself using the mouse. A window will open.
-   **Press 'Enter'** to finalize.
-   **Press 'c'** to clear.
-   **Press 'q'** to quit.

```bash
python3 src/main.py --image <path_to_your_image> --init manual
```

**4. OpenCV Real-time Visualization (`--realtime`)**

This real-time mode uses OpenCV and works for both serial and parallel evolution. You can combine it with manual initialization.

```bash
# Manually draw a contour and then watch it evolve in parallel
python3 src/main.py --image <path_to_your_image> --init manual --realtime --mode parallel
```

## Performance Benchmarking

To reproduce the performance evaluation, run:
```bash
python3 run_benchmarks.py
```
This script runs both serial and parallel modes on all test images, logs the times to `results/performance_log.csv`, and generates comparison plots. The final analysis can be found in `results/performance_report.md`.
