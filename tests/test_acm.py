import unittest
import os
import sys
import cv2
import numpy as np

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
from src.utils import preprocess_image, initialize_snake, visualize_and_save
from src.serial_acm import run_serial

class TestACM(unittest.TestCase):

    def setUp(self):
        """Set up for the tests."""
        self.test_image_path = "data/circle_256.png"
        self.output_dir = "results"
        self.output_path = os.path.join(self.output_dir, "test_output.png")

        # Ensure the test image exists
        if not os.path.exists(self.test_image_path):
            # This check is important because the test depends on the data file
            # We create a dummy image if it's missing, to allow tests to run.
            print(f"Test image not found at {self.test_image_path}, creating a dummy image.")
            dummy_img = np.zeros((256, 256), dtype=np.uint8)
            cv2.circle(dummy_img, (128, 128), 50, (255, 255, 255), -1)
            cv2.imwrite(self.test_image_path, dummy_img)


        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_serial_acm_runs_without_error(self):
        """
        Tests that the full serial ACM pipeline runs without crashing
        and produces an output file.
        """
        # 1. Preprocess
        processed_image, original_image = preprocess_image(self.test_image_path)
        self.assertIsNotNone(processed_image)
        self.assertIsNotNone(original_image)

        # 2. Initialize Snake
        initial_snake = initialize_snake(processed_image.shape, config.N_POINTS)
        self.assertEqual(initial_snake.shape, (config.N_POINTS, 2))

        # 3. Run Serial ACM
        final_snake = run_serial(processed_image, initial_snake)
        self.assertEqual(final_snake.shape, (config.N_POINTS, 2))

        # 4. Visualize and Save
        visualize_and_save(original_image, final_snake, self.output_path)

        # 5. Assert output exists
        self.assertTrue(os.path.exists(self.output_path), "Output file was not created.")

if __name__ == '__main__':
    unittest.main()
