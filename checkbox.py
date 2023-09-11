import os
import cv2
import numpy as np
from PIL import Image

# Specify the path to the 'dataset' folder
dataset_folder = 'dataset'

# List all files in the 'dataset' folder
image_files = [os.path.join(dataset_folder, filename) for filename in os.listdir(dataset_folder) if filename.endswith('.png')]

# Process each image in the 'dataset' folder
for image_file in image_files:
    # Read the image into an array
    image_array = cv2.imread(image_file)
    
    # Check the array type
    image_type = type(image_array)
    print("Array Type:", image_type)  # Output: numpy.ndarray

    # Convert the image to grayscale
    gray_scale_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Image thresholding using Otsu's method
    _, img_bin = cv2.threshold(gray_scale_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the binary image
    img_bin = 255 - img_bin

    # Display the binary image
    Image.fromarray(img_bin).show()

    # Set minimum width to detect horizontal lines
    line_min_width = 13

    # Create a kernel to detect horizontal lines
    kernel_h = np.ones((1, line_min_width), np.uint8)

    # Create a kernel to detect vertical lines
    kernel_v = np.ones((line_min_width, 1), np.uint8)

    # Apply the horizontal kernel to the binary image
    img_bin_horizontal = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_h)

    # Apply the vertical kernel to the binary image
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_v)

    # Combine the horizontal and vertical line images
    img_bin_final = img_bin_horizontal | img_bin_v

    # Perform connected components analysis to label regions
    _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

    # Draw rectangles around detected regions on the original color image
    for x, y, w, h, area in stats[2:]:
        cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the processed image with rectangles drawn around text regions
    Image.fromarray(image_array).show()
