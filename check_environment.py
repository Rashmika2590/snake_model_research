import matplotlib
import sys
import platform

print("--- GUI Environment Check ---")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python Version: {sys.version}")
print(f"Matplotlib Version: {matplotlib.__version__}")

# --- Check Matplotlib Backend ---
print("\n[Step 1] Checking Matplotlib backend...")
try:
    current_backend = matplotlib.get_backend()
    print(f"Current Matplotlib Backend: {current_backend}")

    # Attempt to create a plot
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(2, 2))
    plt.plot([1, 2])
    print("SUCCESS: Matplotlib was able to create a figure object.")
    plt.close(fig)
except Exception as e:
    print(f"ERROR: Could not create a Matplotlib figure. Details:\n{e}")

# --- Check OpenCV GUI ---
print("\n[Step 2] Checking OpenCV GUI capabilities...")
try:
    import cv2
    print(f"OpenCV Version: {cv2.__version__}")
    cv2.namedWindow("OpenCV_Test", cv2.WINDOW_NORMAL)
    cv2.destroyAllWindows()
    print("SUCCESS: OpenCV was able to create and destroy a window.")
except Exception as e:
    print(f"ERROR: OpenCV could not create a window. This is common in headless environments. Details:\n{e}")

print("\n--- End of Check ---")
print("Please copy and paste this entire output in your reply.")
