# Importing necessary libraries
from __future__ import division
import cv2
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import glob
import mahotas
import pandas as pd
import os


# Importing necessary libraries
Testpath = glob.glob(r"C:\Users\Varsha\Desktop\Test\*.png")  # Path for test images
Trainpath = glob.glob(r"C:\Users\Varsha\Desktop\Traindata\*.png")  # Path for training images



def color(image):
    """
    Function to calculate the color histogram of an image.
    
    Parameters:
    image (ndarray): Input image in BGR format.
    
    This function resizes the image, calculates the histogram for each color channel (B, G, R),
    and returns the histogram for the color features.
    
    Returns:
    histr (ndarray): The histogram for one of the color channels.
    """
    bins = 8  # Number of bins for the histogram
    small = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)  # Resize the image to 10% of the original size
    color = ('b', 'g', 'r')  # Color channels: Blue, Green, Red

    # Calculate and return the histogram for each color channel
    for i, col in enumerate(color):
        histr = cv2.calcHist([small], [i], None, [bins], [0, 256])  # Calculate histogram for current channel
        '''
        var1 = input("Do you want to see histogram plot and color values?:\n")
        if var1 == 'y' or var1 == 'Y':
            plt.plot(histr, color=col)
            plt.xlim([0, bins])
            plt.xlabel("Total number bins")
            plt.ylabel("Total number of pixels")
            plt.title("HISTOGRAM PLOT")
            plt.show()
            print("NUMBER OF PIXELS IN ALL {} BINS:\n".format(bins))
            print(histr)
            return histr
        else:
            return histr
        '''
        return histr  

def texture(image):
    """
    Function to calculate texture features using Haralick texture descriptors.
    
    Parameters:
    image (ndarray): Input image in BGR format.
    
    This function converts the image to grayscale and calculates Haralick texture features
    using Mahotas. It returns the mean of the Haralick feature vector.
    
    Returns:
    harlk (ndarray): Array containing Haralick texture features.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    harlk = mahotas.features.haralick(gray).mean(axis=0)  # Calculate Haralick texture features
    '''
    var2 = input("Do you want to see texture moments?:\n")
    if var2 == 'y' or var2 == 'Y':
        print("13 HARALICK TEXTURE FEATURE VECTORS:\n\n")
        print("Angular Second Moment:\t\t", harlk[0])
        print("Contrast:\t\t\t", harlk[1])
        print("Correlation:\t\t\t", harlk[2])
        print("Sum of Squares-Variance:\t", harlk[3])
        print("Inverse Difference moments:\t", harlk[4])
        print("Sum average:\t\t\t", harlk[5])
        print("Sum Variance:\t\t\t", harlk[6])
        print("Sum Entropy:\t\t\t", harlk[7])
        print("Entropy:\t\t\t", harlk[8])
        print("Difference variance:\t\t", harlk[9])
        print("Difference entropy:\t\t", harlk[10])
        print("Info.Measure of correlation 1:\t", harlk[11])
        print("Info.Measure of correlation 2:\t", harlk[12])
        return harlk
    else:
        return harlk
    '''
    return harlk  

def shape(image):
    """
    Function to calculate shape features using Hu moments.
    
    Parameters:
    image (ndarray): Input image in BGR format.
    
    This function converts the image to grayscale and calculates Hu moments,
    which are invariant to scale, rotation, and translation. It returns the Hu moments.
    
    Returns:
    momnt (ndarray): Array of Hu moment values.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    momnt = cv2.HuMoments(cv2.moments(image)).flatten()  # Calculate Hu moments
    '''
    var3 = input("Do you want to see shape moments?:\n")
    if var3 == 'y' or var3 == 'Y':
        print(momnt)
        return momnt
    else:
        return momnt
    '''
    return momnt  



# Functions to display, overlay mask, and crop image

green = (0, 0, 255)  # Color for drawing ellipses (red in BGR format)

def show(image):
    """
    Function to display an image using Matplotlib.
    
    Parameters:
    image (ndarray): Input image to be displayed.
    
    This function displays the input image in a figure of size 10x10.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

def overlay_mask(mask, image):
    """
    Function to overlay a mask on the input image.
    
    Parameters:
    mask (ndarray): Binary mask where detected regions are white.
    image (ndarray): Original image on which the mask will be overlaid.
    
    This function converts the mask to RGB format and overlays it on the original image.
    
    Returns:
    img (ndarray): Image with the mask overlaid.
    """
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)  # Convert the mask to RGB
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)  # Overlay the mask onto the image
    return img  

def crop(image):
    """
    Function to crop an image based on white pixels in a binary image.
    
    Parameters:
    image (ndarray): Input image in BGR format.
    
    This function converts the image to grayscale, applies binary thresholding, and crops
    the image based on the white pixels in the binary image.
    """
    crop_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    res = cv2.resize(crop_img, (0, 0), fx=0.3, fy=0.3)  # Resize the image to 30% of its original size
    _, im_bw = cv2.threshold(res, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Apply binary thresholding
    ret, thresh = cv2.threshold(res, 195, 255, 0)  # Apply a second threshold
    thresh = cv2.bitwise_not(thresh)  # Invert the binary image (swap black and white)
    white_pixels = np.array(np.where(thresh == 255))  # Find white pixels
    s1, e1, s2, e2 = white_pixels[0][0], white_pixels[0][-1], min(white_pixels[1]), max(white_pixels[1])  # Crop bounds
    cv2.imshow('bw', thresh)  # Display the binary image
    cv2.waitKey()
    cropped = thresh[s1:e1, s2:e2]  # Crop the image using calculated bounds
    cv2.imshow("cropped", cropped)  # Display the cropped image
    cv2.waitKey()

def circle_contour(image, contour):
    """
    Function to draw an ellipse around the largest contour.
    
    Parameters:
    image (ndarray): Input image on which the contour will be drawn.
    contour (ndarray): The largest contour detected.
    
    This function draws an ellipse around the contour and returns the modified image.
    
    Returns:
    image_with_ellipse (ndarray): Image with the ellipse drawn around the contour.
    """
    image_with_ellipse = image.copy()  # Create a copy of the image
    ellipse = cv2.fitEllipse(contour)  # Fit an ellipse to the contour
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)  # Draw the ellipse on the image
    return image_with_ellipse  


# Function to find the biggest contour in the image
def find_biggest_contour(image):
    """
    Function to find and return the biggest contour in the input image.
    
    Parameters:
    image (ndarray): Binary image with contours.
    
    This function finds all the contours in the image, selects the largest one based on area,
    and creates a mask that highlights the biggest contour.
    
    Returns:
    biggest_contour (ndarray): The largest contour detected.
    mask (ndarray): Binary mask with the largest contour highlighted.
    """
    image = image.copy()  # Make a copy of the image to avoid modifying the original
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]  # List contour areas
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]  # Find the largest contour by area
    mask = np.zeros(image.shape, np.uint8)  # Create an empty mask
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)  # Draw the largest contour on the mask
    return biggest_contour, mask  

# Function to detect and highlight the leaf in the image
def find_leaf(image):
    """
    Function to detect the leaf in the image and highlight it.
    
    Parameters:
    image (ndarray): Input image in BGR format.
    
    This function converts the image to HSV, applies a color mask to detect green (the leaf), 
    and then finds the largest contour, overlays the mask, and draws an ellipse around the leaf.
    
    Returns:
    bgr (ndarray): Image with the leaf detected and highlighted in BGR format.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
    max_dimension = max(image.shape)  # Find the largest dimension of the image
    scale = 700 / max_dimension  # Calculate the scaling factor to resize the image
    image = cv2.resize(image, None, fx=scale, fy=scale)  # Resize the image
    image_blur_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # Convert the image to HSV
    min_green = np.array([40, 40, 40])  # Define the lower bound for green color
    max_green = np.array([70, 255, 255])  # Define the upper bound for green color
    mask = cv2.inRange(image_blur_hsv, min_green, max_green)  # Create a mask for detecting green
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # Define a morphological kernel
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)  # Apply morphological closing to fill gaps
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, k)  # Apply morphological opening to remove noise
    big_leaf_contour, mask_leaf = find_biggest_contour(mask_clean)  # Find the largest contour (the leaf)
    overlay = overlay_mask(mask_clean, image)  # Overlay the mask onto the original image
    circled = circle_contour(overlay, big_leaf_contour)  # Draw an ellipse around the leaf
    show(circled)  # Display the result
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)  # Convert the result back to BGR format
    return bgr  

# Function to capture images from the camera
def camtrig():
    """
    Function to trigger the camera, capture images, and save them when a leaf is detected.
    
    This function continuously captures frames from the camera. If the space bar is pressed, 
    it checks for a leaf (based on a green color mask) and saves the frame if a leaf is detected.
    
    Returns:
    img_name (str): The file path of the last saved image.
    """
    cam = cv2.VideoCapture(0)  # Open the default camera
    cv2.namedWindow("test")  # Create a window for displaying the video feed
    d = 0  # Counter for naming saved images
    while True:
        ret, frame = cam.read()  # Capture frame-by-frame
        cv2.imshow("test", frame)  # Display the captured frame
        if not ret:
            break
        k = cv2.waitKey(1)  # Wait for a key press

        if k % 256 == 27:
            # If 'ESC' key is pressed, exit the loop
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # If 'SPACE' key is pressed, check for a leaf
            if ret:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert the frame to HSV
                lower = np.array([40, 40, 40])  # Lower bound for green color
                upper = np.array([70, 255, 255])  # Upper bound for green color
                mask = cv2.inRange(hsv, lower, upper)  # Create a mask for detecting green
                cv2.imshow("image", mask)  # Display the mask

                if mask.any() == True:
                    # If a leaf is detected, save the image
                    img_name = "Testpath/test{}.png".format(d)
                    cv2.imwrite(img_name, frame)  # Save the captured frame
                    print("{} written!".format(img_name))
                else:
                    print("Oops. Where is the leaf ?!!")
            d += 1  # Increment the counter for the next image

    cam.release()  # Release the camera
    cv2.destroyAllWindows()  # Close the window
    return img_name  

# Function to extract features from training data
def Data(path):
    """
    Function to extract features (color, texture, and shape) from images in the specified path.
    
    Parameters:
    path (list): List of file paths to the training images.
    
    This function reads each image from the path, extracts color, texture, and shape features,
    and stores them in a DataFrame along with labels (1 for diseased, 0 for healthy).
    
    Returns:
    DF (DataFrame): Pandas DataFrame containing the features and labels.
    label (list): List of labels corresponding to each image.
    """
    Trainpath1 = os.listdir(r"C:\Users\Varsha\Desktop\Traindata")  # Get list of image files in the training data
    label = []  # Initialize an empty list for storing labels

    # Assign labels based on file name ('d' for diseased, 'h' for healthy)
    for file in Trainpath1:
        y = file.split('.')[0]
        if y == 'd':
            label.append(1)
        elif y == 'h':
            label.append(0)

    q = 0  # Counter for sample numbers
    d = {'Sample': [], 'Label': label}  # Initialize a dictionary to store feature values and labels

    # Initialize lists in the dictionary for storing color, texture, and shape features
    for i in range(8):
        d['value' + str(i)] = []
    for j in range(13):
        d['value' + str(8 + j)] = []
    for k in range(7):
        d['value' + str(21 + k)] = []

    # Iterate over each image path and extract features
    for img in path:
        n = cv2.imread(img)  # Read the image
        colorCall = color(n)  # Extract color features
        texCall = texture(n)  # Extract texture features
        shapeCall = shape(n)  # Extract shape features
        d['Sample'].append(q)  # Add sample number to the dictionary

        # Add the extracted features to the dictionary
        for i in range(8):
            d['value' + str(i)].append(colorCall[i][0])
        for i in range(13):
            d['value' + str(8 + i)].append(texCall[i])
        for i in range(7):
            d['value' + str(21 + i)].append(shapeCall[i])
        q += 1  # Increment the sample counter

    DF = pd.DataFrame(d)  # Convert the dictionary to a Pandas DataFrame
    return DF, label  

# Function to perform KNN classification
def KNNlazy(frame, label):
    """
    Function to perform K-Nearest Neighbors (KNN) classification on the provided data.
    
    Parameters:
    frame (DataFrame): The feature matrix (input data).
    label (list): The labels corresponding to each sample.
    
    This function splits the data into training and testing sets, trains a KNN classifier,
    and predicts labels for the test data. It prints the accuracy of the classifier.
    
    Returns:
    y_pred (ndarray): Predicted labels for the test data.
    """
    X_train, X_test, y_train, y_test = train_test_split(frame, label, test_size=0.3)  # Split the data into training and testing sets
    knn = KNeighborsClassifier(n_neighbors=5)  # Initialize a KNN classifier with 5 neighbors
    knn.fit(X_train, y_train)  # Train the classifier on the training data
    y_pred = knn.predict(X_test)  # Predict labels for the test data
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))  # Print the accuracy of the classifier
    print(X_train, y_train, y_test, X_test)  # Print the training and testing data
    return y_pred  

# Example usage

p = 0  # Initialize counter for saving CSV files

traindata, label = Data(Trainpath)  # Extract features from the training data
traindata.to_csv("TrainData{}.csv".format(p))  # Save the extracted features to a CSV file

knntr = KNNlazy(traindata, label)  # Perform KNN classification on the training data
print(knntr)  # Print the predicted labels
