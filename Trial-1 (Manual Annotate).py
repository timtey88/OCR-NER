import os
import streamlit as st
import easyocr
import cv2
import csv
from PIL import Image
import pandas as pd
import numpy as np

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to detect boxes in a given region of interest (ROI)
def detect_boxes(roi):
    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale ROI
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using Canny
    edged = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and extract the detected boxes
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            (x, y, w, h) = cv2.boundingRect(contour)
            boxes.append((x, y, x + w, y + h))
    
    return boxes

# Function to read text in regions of interest (ROI)
def read_text_in_order(image_path, regions_of_interest):
    # Read the image
    image = cv2.imread(image_path)

    # Initialize empty dictionary to store text and labels
    extracted_info = {}

    # Create a copy of the image to draw rectangles on
    image_with_rectangles = image.copy()

    # Initialize Controlled Items Indicator
    controlled_items = {}

    for x1, y1, x2, y2, roi_type, label in regions_of_interest:
        roi = image[y1:y2, x1:x2]  # Crop the region of interest

        if roi_type == 'box':
            # Detect boxes in the ROI
            boxes = detect_boxes(roi)
            
            # Draw rectangles around detected boxes on the original color image
            for (x, y, x_end, y_end) in boxes:
                cv2.rectangle(image_with_rectangles, (x1 + x, y1 + y), (x1 + x_end, y1 + y_end), (0, 255, 0), 2)
        else:
            # Perform OCR on the cropped region using EasyOCR
            results = reader.readtext(roi)

            # Extract text from the EasyOCR results
            text = " ".join([result[1] for result in results])

            # Store extracted text based on label
            if label.lower() == 'controlled_items_indicator':
                controlled_items[label.capitalize()] = text
            else:
                extracted_info[label.capitalize()] = text

            # Draw a green rectangle around the region of interest
            cv2.rectangle(image_with_rectangles, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Combine extracted info with Controlled Items Indicator
    extracted_info.update(controlled_items)

    return extracted_info, image_with_rectangles

# Streamlit UI code
st.title("Image OCR and NER")

# Select a directory containing PNG images
image_dir = os.path.join(os.getcwd(), "dataset")  # Replace with your dataset folder path
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

if not image_files:
    st.warning("No PNG images found in the selected directory. Please upload PNG images.")
else:
    # Select an image from the directory
    selected_image = st.selectbox("Select an Image", image_files)

    # Display the selected image
    image_path = os.path.join(image_dir, selected_image)
    st.image(image_path, caption="Selected Image", use_column_width=True)

    # Define regions of interest (x1, y1, x2, y2, type, label) for text extraction
    regions_of_interest = [
        [510, 42, 628, 68, 'text', 'date'],
        [512, 72, 626, 94, 'text', 'flight_number'],
        [188, 234, 336, 250, 'text', 'name'],
        [76, 346, 92, 360, 'box', 'no_goods'],
        [74, 382, 90, 400, 'box', 'goods'],
        [76, 444, 90, 460, 'box', 'tobacco'],
        [74, 464, 90, 480, 'box', 'alcohol'],
        [76, 496, 90, 514, 'box', 'medical'],
        [74, 518, 88, 532, 'box', 'others'],
        [308, 442, 458, 456, 'text', 'tobacco'],
        [310, 474, 458, 488, 'text', 'alcohol'],
        [308, 492, 458, 510, 'text', 'medical'],
        [310, 510, 458, 530, 'text', 'others'],
        [76, 594, 90, 612, 'box', 'no_goods'],
        [76, 628, 92, 642, 'box', 'goods'],
        [76, 692, 90, 708, 'box', 'tobacco'],
        [76, 712, 92, 730, 'box', 'alcohol'],
        [76, 744, 92, 760, 'box', 'medical'],
        [76, 766, 90, 780, 'box', 'others'],
        [312, 688, 458, 704, 'text', 'tobacco'],
        [308, 722, 458, 736, 'text', 'alcohol'],
        [310, 740, 458, 756, 'text', 'medical'],
        [310, 756, 460, 776, 'text', 'others']
    ]

    # Check if the button is clicked
    if st.button("Extract Information"):
        # Step 1: Read Text from Image within defined regions
        extracted_info, image_with_rectangles = read_text_in_order(image_path, regions_of_interest)

        # Step 2: Display the extracted information
        st.header("Extracted Information:")
        st.write(f"Name: {extracted_info.get('Name', '-')}")
        st.write(f"Travelling Date: {extracted_info.get('Date', '-')}")
        st.write(f"Flight Number: {extracted_info.get('Flight_number', '-')}")
        st.write("Controlled Items Indicator:")
        for label in ['Tobacco', 'Alcohol', 'Medical', 'Others']:
            st.write(f"{label}: {extracted_info.get(label, '-')}")

        # Display the image with highlighted regions
        st.image(image_with_rectangles, caption="Image with Highlighted Regions", use_column_width=True)

        # Step 3: Write data to a CSV file
        csv_file_path = 'extracted_data.csv'
        with open(csv_file_path, mode='w', newline='') as csv_file:
            fieldnames = ['Label', 'Extracted Text']
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            for label, text in extracted_info.items():
                csv_writer.writerow({'Label': label, 'Extracted Text': text})

        # Step 4: Display the CSV table
        st.header("Extracted Data in CSV Format:")
        extracted_data = pd.read_csv(csv_file_path)
        st.write(extracted_data)

        # Optionally, provide a download link for the CSV file
        st.markdown(f'[Download CSV](./{csv_file_path})')


#streamlit run /Users/tt/workspace/mindhive-ocr/mindhive-ocr.py