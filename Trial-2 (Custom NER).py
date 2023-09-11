import os
import cv2
import pytesseract
import streamlit as st
import pandas as pd
import numpy as np
import spacy
import csv

# Step 1: Image Preprocessing
def preprocess_image(uploaded_image):
    # Convert the uploaded file object to an OpenCV image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Apply necessary preprocessing techniques
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    denoised_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
    
    return denoised_image

# Step 2: OCR Application
def extract_text_from_image(uploaded_image):
    preprocessed_image = preprocess_image(uploaded_image)
    text = pytesseract.image_to_string(preprocessed_image)
    return text

# Step 3: Text Data Cleaning
def clean_text(text):
    cleaned_text = text.replace('\n', ' ').strip()
    return cleaned_text

# Step 4: Named Entity Recognition (NER)
def perform_ner(text):
    nlp = spacy.load('./output/model-best')
    doc = nlp(text)
    
    entities = {}
    for ent in doc.ents:
        entities[ent.label_] = ent.text
    
    return entities

# Step 5: Data Storage (Using Pandas for simplicity)
def store_data(entities):
    df = pd.DataFrame([entities])
    df.to_csv('extracted_data.csv', index=False)

# Step 6: UI/UX using Streamlit
def main():
    st.title("Image Form Data Extraction")
    uploaded_images = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_images:
        for uploaded_image in uploaded_images:
            st.header(f"Extracted Data from {uploaded_image.name}")
            
            # Extract text from the uploaded image
            text = extract_text_from_image(uploaded_image)
            cleaned_text = clean_text(text)
            
            # Display the extracted text
            st.subheader("Cleaned Text:")
            st.write(cleaned_text)
            
            # Perform NER
            entities = perform_ner(cleaned_text)
            
            # Display the extracted data
            st.subheader("Named Entities:")
            st.write(entities)

            # Step 2: Display the extracted information
            st.header("Extracted Information:")
            for label, text in entities.items():
                st.write(f"{label}: {text}")

            # Step 3: Write data to a CSV file
            csv_file_path = 'extracted_data.csv'
            with open(csv_file_path, mode='w', newline='') as csv_file:
                fieldnames = ['Label', 'Extracted Text']
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()

                for label, text in entities.items():
                    csv_writer.writerow({'Label': label, 'Extracted Text': text})

            # Step 4: Display the CSV table
            st.header("Extracted Data in CSV Format:")
            extracted_data = pd.read_csv(csv_file_path)
            st.write(extracted_data)

            # Optionally, provide a download link for the CSV file
            st.markdown(f'[Download CSV](./{csv_file_path})')

if __name__ == "__main__":
    main()

# streamlit run /Users/tt/workspace/mindhive-ocr/trial-2.py
