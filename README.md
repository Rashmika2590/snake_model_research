# Active Contour Model (Snake) for Image Segmentation

This project implements and evaluates an Active Contour Model (ACM) for image segmentation. 
It provides both a standard serial implementation and a highly optimized parallel version using Numba. 
The application supports standard execution and interactive modes for manual contour initialization and real-time visualization of the snake's evolution.

## Project Structure

- `/src`: Source code for the ACM implementations, utilities, and configuration.
- `/tests`: Test scripts for ensuring code correctness.
- `/data`: Sample images for testing and benchmarking.
- `/results`: Output images, performance logs, and plots.
- `run_benchmarks.py`: A script to automate the performance evaluation.

## Setup

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Generate Test Data:**
    ```bash
    python3 tests/generate_test_images.py
    ```

## Usage

### Serial Execution
Run the standard serial version:
```bash
python3 src/main.py --image data/circle_256.png --mode serial
