import streamlit as st
import zipfile
import os
import tempfile
import cv2
import numpy as np
from pathlib import Path
import dlib
from deepface import DeepFace
import shutil

DLIB_1 = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
DLIB_2 = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
           

# Dlib face detection function
def dlib_face_detection(known_face_path, input_folder, output_folder):
    # Initialize dlib's face detector and recognition model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    # Load known face
    known_image = cv2.imread(known_face_path)
    known_gray = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)
    known_faces = detector(known_gray)
    if not known_faces:
        return "No face found in the reference image."

    known_shape = predictor(known_gray, known_faces[0])
    known_face_encoding = np.array(face_rec_model.compute_face_descriptor(known_image, known_shape))

    results = []
    for format in ["*.jpeg", "*.jpg", "*.png"]:
        for photo_path in Path(input_folder).glob(format):
            image = cv2.imread(str(photo_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if not faces:
                results.append(f"No face found in {photo_path.name}")
                continue

            for face in faces:
                shape = predictor(gray, face)
                face_encoding = np.array(face_rec_model.compute_face_descriptor(image, shape))

                distance = np.linalg.norm(known_face_encoding - face_encoding)
                if distance < 0.6:  # Adjust threshold as needed
                    shutil.copy(str(photo_path), os.path.join(output_folder, photo_path.name))
                    results.append(f"Match found: {photo_path.name}")
                else:
                    results.append(f"No match: {photo_path.name}")

    return "\n".join(results)

# DeepFace detection function
def deepface_detection(known_face_path, input_folder, output_folder):
    results = []
    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_image_path = os.path.join(input_folder, image_file)
            try:
                result = DeepFace.verify(img1_path=known_face_path, 
                                         img2_path=input_image_path, 
                                         model_name='VGG-Face', 
                                         distance_metric='cosine')

                if result['verified']:
                    shutil.copy(input_image_path, os.path.join(output_folder, image_file))
                    results.append(f"Match found: {image_file}")
                else:
                    results.append(f"No match: {image_file}")
            except Exception as e:
                results.append(f"Error processing {image_file}: {str(e)}")
    
    return "\n".join(results)

# Streamlit app
st.title("Face Detection App")

st.info("""Disclaimer: This app uses the DeepFace library for face recognition. The accuracy of the face recognition models may vary based on the input images and conditions. We donot guarantee the accuracy of the results. 
        
We use temporary directories to store the uploaded images and results. The images are deleted after the session ends. Hence, the images are not stored or used for any other purposes.

This app supports JPEG, JPG, and PNG image formats.
The face detection process may take some time depending on the number of images and the selected model. Please be patient.
       """)


# Model selection
model = st.radio("Select face detection model:", ("DeepFace", "dlib"))

# File upload
uploaded_zip = st.file_uploader("Upload ZIP file containing images (max 25 images)", type="zip")
reference_image = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png"])

if uploaded_zip and reference_image:
    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_folder = os.path.join(tmpdirname, "input")
        output_folder = os.path.join(tmpdirname, "output")
        os.makedirs(input_folder)
        os.makedirs(output_folder)

        # Save reference image
        ref_image_path = os.path.join(tmpdirname, "reference.jpg")
        with open(ref_image_path, "wb") as f:
            f.write(reference_image.getbuffer())

        # Extract ZIP file
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(input_folder)

        # Check number of images
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) > 35:
            st.error("Error: More than 35 images found in the ZIP file.")
        else:
            # Run face detection
            if model == "DeepFace":
                results = deepface_detection(ref_image_path, input_folder, output_folder)
            else:  # dlib
                results = dlib_face_detection(ref_image_path, input_folder, output_folder)

            st.text_area("Detection Results", results, height=250)

            # Display matched images
            matched_images = os.listdir(output_folder)
            if matched_images:
                st.subheader("Matched Images")
                for img in matched_images:
                    st.image(os.path.join(output_folder, img), caption=img, use_column_width=True)
            else:
                st.info("No matching images found.")


