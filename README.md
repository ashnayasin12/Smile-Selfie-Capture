# Smile Selfie Capture  

Smile Selfie Capture is an intelligent and interactive computer vision project that uses HAAR Cascade Classifiers to automatically detect smiles and capture selfies in real-time. This project leverages the power of OpenCV and Python to create an intuitive and fun way to take selfies without pressing a button—just smile, and let the system do the rest!  

## Features  
- Smile Detection: Automatically captures selfies when a smile is detected, even when teeth aren't visible.  
- Face Detection Overlays: Rectangular overlays for face and smile detection appear in the live video feed but are excluded from saved images.  
- Image Filters: Customizable filters like Grayscale, Sepia, and Blur can be applied to captured selfies based on user preference.  
- Voice Feedback: Provides real-time audio confirmation when a selfie is successfully captured, enhancing user interactivity.  
- Accurate Detection: Utilizes pre-trained HAAR Cascade models for precise and efficient detection of faces and smiles.  
- Real-Time Processing: Ensures smooth and fast video feed analysis and photo capture.  

## Tools & Technologies  
- Python: Core programming language for implementation.  
- OpenCV: Library for computer vision tasks, including face and smile detection.  
- HAAR Cascade Classifiers: Pre-trained models for face (`haarcascade_frontalface_default.xml`) and smile (`haarcascade_smile.xml`) detection.  
- Pyttsx3: Python library for text-to-speech functionality to provide voice feedback.  

## How It Works  
1. The program accesses your system's camera to display a real-time video feed.  
2. Using HAAR Cascade Classifiers, it detects faces and smiles in the video stream.  
3. When a smile is detected, the system automatically captures a selfie.  
4. Provides audio feedback to confirm that the selfie has been successfully captured.  
5. The captured selfie can optionally have filters applied before being saved.  

## Future Scope  
- Integration with social media for instant sharing of selfies.  
- Enhanced detection using deep learning-based models for even greater accuracy.  
- A web-based version for more accessibility.  

## Why Smile Selfie Capture?  
Smile Selfie Capture provides a unique and engaging way to take selfies, perfect for fun applications or as an interactive tech demo for computer vision capabilities. It’s simple, efficient, and showcases the power of real-time detection, automation, and user interactivity using OpenCV.  
>>>>>>> 5761d0b9e6a6f217d742e8317a68e05c09875443
