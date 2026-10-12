import numpy as np
import cv2

# Крок 5: Зафіксуйте випадкові стани
np.random.seed(42)

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)