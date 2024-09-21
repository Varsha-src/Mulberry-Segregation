# Importing necessary libraries
from __future__ import division  # Ensures compatibility between Python 2 and 3 for division
import cv2  # OpenCV library for image processing
import numpy as np  # Numpy for numerical operations
from matplotlib import pyplot as plt  # Matplotlib for plotting histograms or images
import mahotas  # Mahotas for calculating Haralick texture features
import glob  # Glob for file path matching

#%%
# Capturing images through camera and writing them to a folder

def color(image):
    """
    Function to calculate the color histogram of an image in HSV color space.
    
    Parameters:
    image (ndarray): Input image in BGR format.
    
    This function converts the image to HSV, calculates the histogram in 3D (for H, S, V channels),
    and normalizes it. It prints the flattened histogram.
    """
    bins = 8  # Number of bins for the histogram
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert the image to HSV color space
    histg = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])  # Calculate 3D histogram
    cv2.normalize(histg, histg)  # Normalize the histogram
    print(histg.flatten())  # Print the flattened histogram

# Function to calculate texture using Haralick features
def texture(image):
    """
    Function to calculate the Haralick texture features from an image.
    
    Parameters:
    image (ndarray): Input image in BGR format.
    
    This function converts the image to grayscale and computes the Haralick texture features
    using Mahotas. It prints the average of the Haralick feature vectors.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    harlk = mahotas.features.haralick(grey).mean(axis=0)  # Calculate Haralick texture features
    print(harlk)  # Print the texture feature vector

# Function to calculate shape using Hu moments
def shape(image):
    """
    Function to calculate shape features using Hu moments.
    
    Parameters:
    image (ndarray): Input image in BGR format.
    
    This function converts the image to grayscale and computes the Hu moments, which are
    invariant to scale, rotation, and translation. It prints the flattened Hu moments.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    momnt = cv2.HuMoments(cv2.moments(image)).flatten()  # Calculate Hu moments
    print(momnt)  # Print the Hu moments

# Initialize the webcam to capture images
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")  # Create a window to display the video feed
count = 0  # Counter for saved images

while True:
    ret, frame = cam.read()  # Capture frame-by-frame from the webcam
    cv2.imshow("test", frame)  # Display the captured frame

    if not ret:
        break  # Exit if frame capture fails

    k = cv2.waitKey(1)  # Wait for a key press

    if k % 256 == 27:
        # If 'ESC' key is pressed, exit the loop
        print("Closing!")
        break
    elif k % 256 == 32:
        # If 'SPACE' key is pressed, capture the image and process it
        if ret:
            # Convert the captured frame to HSV and create a mask for detecting green areas (representing leaves)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([40, 40, 40])  # Lower bound for green color in HSV
            upper = np.array([70, 255, 255])  # Upper bound for green color in HSV
            mask = cv2.inRange(hsv, lower, upper)  # Create a mask for the green areas
            cv2.imshow("image", mask)  # Display the mask

            # If green area is detected in the mask, save the image
            if mask.any() == True:
                img_name = r"C:\Users\Varsha\Desktop\Major Project\CODES\Camcapture\file{}.png".format(count)
                cv2.imwrite(img_name, frame)  # Save the captured image
                print("{} written!".format(img_name))
            else:
                print("Oops. Where is the leaf ?!!")
        count += 1  # Increment the image counter

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()

# Resizing the captured images and extracting features

# Get paths of all captured images
prepath = glob.glob(r"C:/Users/Varsha/Desktop/Major Project/CODES/Camcapture/*.png")
img = []  # Placeholder for storing images

# Loop through each image path, read the image, resize, and extract features
for img in prepath:
    a = cv2.imread(img)  # Read the image
    smallimg = cv2.resize(a, (0, 0), fx=1, fy=1)  # Resize the image (currently, no scaling applied)
    cv2.imshow('re', smallimg)  # Display the resized image
    
    # Call the feature extraction functions
    colorCall = color(smallimg)  # Extract color features
    texCall = texture(smallimg)  # Extract texture features
    shapeCall = shape(smallimg)  # Extract shape features

#%%

# Feature extraction from the captured and processed images
# The color, texture, and shape features are already extracted in the previous loop.
