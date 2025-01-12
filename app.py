# import cv2
# import os
# import time

# # Initialize video capture
# video = cv2.VideoCapture(0)

# # Load Haar Cascade Classifiers
# faceCascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
# smileCascade = cv2.CascadeClassifier("haarcascade/haarcascade_smile.xml")

# # Ensure the directory exists
# directory = r'C:\Users\raven\Desktop\images'  # Adjust for your OS
# if not os.path.exists(directory):
#     os.makedirs(directory)

# cnt = 1
# last_capture_time = 0  # Track the last capture time
# capture_cooldown = 2  # Cooldown period in seconds

# while True:
#     success, img = video.read()
    
#     # Check if image was captured successfully
#     if not success or img is None:
#         print("Error: Failed to capture image or image is None.")
#         break
    
#     # Process image
#     grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=6)
#     keyPressed = cv2.waitKey(1)

#     for (x, y, w, h) in faces:
#         print("Face detected.")
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw face rectangle

#         # Define the region of interest (ROI) for smile detection
#         roi_gray = grayImg[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]

#         # Adjusted smile detection parameters
#         smiles = smileCascade.detectMultiScale(
#             roi_gray, 
#             scaleFactor=1.8, 
#             minNeighbors=35,  # Increase this to reduce false positives
#             minSize=(25, 25),  # Minimum size for a smile
#             maxSize=(200, 200)  # Maximum size for a smile
#         )

#         if len(smiles) > 0:
#             print("Smile detected.")

#             # Only capture if enough time has passed since the last capture
#             current_time = time.time()
#             if current_time - last_capture_time > capture_cooldown:
#                 path = os.path.join(directory, f'image{cnt}.jpg')
#                 if cv2.imwrite(path, img):
#                     print(f"Image {cnt} saved successfully at {path}")
#                 else:
#                     print("Error: Failed to save image.")

#                 last_capture_time = current_time
#                 cnt += 1

#                 # Optional: break to avoid capturing too many images quickly
#                 if cnt >= 2:    
#                     break

#     # Display the live video
#     cv2.imshow('live video', img)
#     if keyPressed & 0xFF == ord('q'):
#         break

# # Release resources
# video.release()                                  
# cv2.destroyAllWindows()

import os
import cv2
import numpy as np
import time
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load Haar Cascade Classifiers
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_smile.xml')

# Global variable to store selected filter
selected_filter = None

# Ensure that the directory exists
output_dir = r'C:\Users\raven\Desktop\image\\'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the directory if it doesn't exist

# Function to apply filters
def apply_filter(image, filter_type):
    if filter_type == "grayscale":
        print("Applying Grayscale Filter")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == "sepia":
        print("Applying Sepia Filter")
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_image = cv2.transform(image, sepia_filter)
        sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
        return sepia_image
    elif filter_type == "blur":
        print("Applying Blur Filter")
        return cv2.GaussianBlur(image, (15, 15), 0)
    else:
        print("No filter applied")
        return image  # No filter applied

# Function to save images
def save_image(image):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f'selfie_filtered_{timestamp}.jpg'
    filepath = os.path.join(output_dir, filename)  # Properly combine path and filename
    try:
        # Save the image
        cv2.imwrite(filepath, image)
        print(f"Filtered selfie saved as '{filepath}'")
        engine.say("Selfie saved!")  # Voice feedback for saving image
        engine.runAndWait()
    except Exception as e:
        print(f"Failed to save image: {e}")

# Start video capture
cap = cv2.VideoCapture(0)
last_saved_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the frame isn't captured

    # Make a copy of the original frame for saving without rectangles
    frame_for_saving = frame.copy()

    # Resize frame for faster face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Detect faces
    faces = face_cascade.detectMultiScale(small_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Scale back face coordinates to the original size
        x, y, w, h = [v * 2 for v in (x, y, w, h)]
        # Draw rectangle around face on the displayed frame only
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect smile within the face
        roi_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)  # Convert to grayscale for smile detection
        smiles = smile_cascade.detectMultiScale(
            roi_gray, 
            scaleFactor=2.1, 
            minNeighbors=30,  # Increase this to reduce false positives
            minSize=(43, 43),  # Minimum size for a smile
            maxSize=(200, 200)  # Maximum size for a smile
        )

        for (sx, sy, sw, sh) in smiles:
            # Draw rectangle around the smile on the displayed frame only
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)

            # Smile detected, cooldown to prevent multiple triggers
            if time.time() - last_saved_time > 5:  # 5-second cooldown
                print("Smile detected! Applying selected filter...")

                # Voice feedback for smile detection
                engine.say("Smile detected!")
                engine.runAndWait()

                # Apply the selected filter to the frame for saving (without rectangles)
                filtered_image = apply_filter(frame_for_saving, selected_filter)

                # Save the filtered image without the rectangles
                save_image(filtered_image)

                last_saved_time = time.time()  # Update last saved time

    # Check for key presses to change filters
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        selected_filter = "grayscale"
        print("Grayscale filter selected")
    elif key == ord('2'):
        selected_filter = "sepia"
        print("Sepia filter selected")
    elif key == ord('3'):
        selected_filter = "blur"
        print("Blur filter selected")

    # Display the resulting frame with real-time filter preview (with rectangles)
    if selected_filter:
        preview_frame = apply_filter(frame.copy(), selected_filter)  # Use a copy for preview
        cv2.imshow('Smile Selfie Capture (Filtered Preview)', preview_frame)
    else:
        cv2.imshow('Smile Selfie Capture', frame)

    # Exit on pressing 'q'
    if key == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
