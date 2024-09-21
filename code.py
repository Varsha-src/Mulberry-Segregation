# Import necessary modules 
from __future__ import division  
import cv2 
from matplotlib import pyplot as plt    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import numpy as np 
import glob 
import mahotas 
import pandas as pd 
import os 
import pickle

# Define file paths for test and train images using glob to search for .png files
Testpath = glob.glob(r"C:\Users\Varsha\Desktp\TestData\test*.png")  # For Test Data Images
Trainpath = glob.glob(r"C:\Users\Varsha\Desktop\TrainData\*.png")  # Path for Train Data Images

#--------------------------------------------------------------Image Acquisition---------------------------------------------------------------------------

def camtrig(): 
    """
    Function to capture images from the camera.
    Captures and saves images if a green object (leaf) is detected.
    """
    # Open the default camera (index 0)
    cam = cv2.VideoCapture(0)
    
    # Create a window to display the captured frames
    cv2.namedWindow("test")
    
    d = 0  # Counter for naming captured images
    
    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()
        
        # Display the captured frame in the window
        cv2.imshow("test", frame)
        
        if not ret:
            # Break the loop if frame capture fails
            break
        
        # Wait for a key press
        k = cv2.waitKey(1)

        # If 'ESC' key is pressed, exit the loop
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        
        # If 'SPACE' key is pressed, process the image
        elif k % 256 == 32:
            if ret:
                # Convert the captured image to HSV color space
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Define lower and upper bounds for green color
                lower = np.array([40, 40, 40])
                upper = np.array([70, 255, 255])
                
                # Create a mask to detect green objects (leaves)
                mask = cv2.inRange(hsv, lower, upper)
                cv2.imshow("image", mask)

                # Check if the mask contains any green pixels (leaf detected)
                if mask.any() == True:
                    # Save the captured frame as a .png image
                    img_name = r"C:\Users\HP\Desktop\ToShow\TestData\Captured\test{}.png".format(d)
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                else:
                    print("Oops. Where is the leaf ?!!")
            
            d += 1  # Increment the counter for the next image

    # Release the camera and close the window
    cam.release()
    cv2.destroyAllWindows()
    
    return img_name  


#-----------------------------------------------------------------------Preprocessing------------------------------------------------------------------

def crop(image): 
    """
    Function to preprocess the input image by cropping and thresholding it.
    Converts the image to grayscale, resizes it, applies binary thresholding, 
    and then crops the region of interest based on white pixels.
    """
    # Convert the image to grayscale
    crop_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image by scaling down the dimensions
    res = cv2.resize(crop_img, (0, 0), fx=0.3, fy=0.3)
    
    # Apply Otsu's thresholding to convert the image to binary (black and white)
    (thresh, im_bw) = cv2.threshold(res, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Apply additional thresholding to highlight specific features
    ret, thresh = cv2.threshold(res, 195, 255, 0)
    
    # Invert the binary image to switch black and white
    thresh = cv2.bitwise_not(thresh)
    
    # Find the white pixels in the thresholded image
    white_pixels = np.array(np.where(thresh == 255))
    
    # Extract coordinates for cropping the image
    s1, e1, s2, e2 = white_pixels[0][0], white_pixels[0][-1], min(white_pixels[1]), max(white_pixels[1])
    
    # Display the binary (black and white) thresholded image
    cv2.imshow('bw', thresh)
    cv2.waitKey()
    
    # Crop the image using the calculated coordinates
    cropped = thresh[s1:e1, s2:e2]
    
    # Display the cropped image
    cv2.imshow("cropped", cropped)
    cv2.waitKey()


        


#--------------------------------------------------------------------Segmentation----------------------------------------------------------------------


def show(img): 
    """
    Function to display an image using matplotlib.
    
    Parameters:
    img (ndarray): The image to be displayed.
    
    This function uses the 'imshow' function from matplotlib to display the image.
    """
    plt.figure(figsize=(10, 10))  # Set the figure size
    plt.imshow(img)  # Display the image

def overlay_mask(mask, img): 
    """
    Function to overlay a mask on the original image.
    
    Parameters:
    mask (ndarray): Binary mask where detected parts are white.
    img (ndarray): Original image on which the mask will be overlaid.
    
    This function converts the grayscale mask to RGB, then overlays it onto the original image with a weighted sum, 
    resulting in an image where the detected parts are highlighted.
    """
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)  # Convert the grayscale mask to RGB
    img = cv2.addWeighted(rgb_mask, 0.5, img, 0.5, 0)  # Overlay the mask onto the original image
    return img  # Return the image with the mask overlaid

def circle_contour(img, contour): 
    """
    Function to draw an ellipse around the biggest contour.
    
    Parameters:
    img (ndarray): Image on which the contour will be drawn.
    contour (ndarray): The contour around which an ellipse will be drawn.
    
    This function draws an ellipse around the given contour and returns the modified image.
    """
    img_with_el = img.copy()  # Create a copy of the image to draw on
    ellipse = cv2.fitEllipse(contour)  # Fit an ellipse around the contour
    cv2.ellipse(img_with_el, ellipse, (0, 0, 255), 2, cv2.LINE_AA)  # Draw the ellipse in red color
    return img_with_el  # Return the image with the ellipse drawn

def find_biggest_contour(img): 
    """
    Function to find the biggest contour in a binary image.
    
    Parameters:
    img (ndarray): Binary image (usually a mask) in which contours will be detected.
    
    This function finds all contours, selects the biggest one based on area, and returns both the biggest contour 
    and a mask containing only this contour.
    """
    img = img.copy()  # Make a copy of the input image
    # Find contours in the image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate contour areas and find the largest one
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]  # Extract the biggest contour
    
    # Create a mask with the biggest contour
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)  # Draw the biggest contour on the mask
    
    return biggest_contour, mask  # Return the biggest contour and the corresponding mask

def find_leaf(img): 
    """
    Function to detect the leaf in the image using morphological operations.
    
    Parameters:
    img (ndarray): Input image in which the leaf needs to be detected.
    
    This function resizes the image, converts it to HSV, applies color thresholding to detect green areas (representing the leaf),
    and uses morphological operations to clean the mask. It then finds the biggest contour (assumed to be the leaf) and overlays 
    the mask on the original image, drawing a circle around the detected leaf.
    
    Returns:
    bgr (ndarray): Final image with the leaf detected, highlighted, and converted back to BGR color space.
    """
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image while maintaining the aspect ratio
    max_dimension = max(img.shape)
    scale = 700 / max_dimension  # Scale factor to resize the image
    img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Convert the image to HSV color space
    img_blur_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Define the range for detecting green color (leaf color)
    min_green = np.array([40, 40, 40])
    max_green = np.array([70, 255, 255])
    
    # Create a binary mask where green areas are white
    mask = cv2.inRange(img_blur_hsv, min_green, max_green)
    
    # Define an elliptical structuring element for morphological operations
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    
    # Apply morphological closing to fill gaps in the mask
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    
    # Apply morphological opening to remove noise from the mask
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, k)
    
    # Find the biggest contour (assumed to be the leaf)
    big_leaf_contour, mask_leaf = find_biggest_contour(mask_clean)
    
    # Overlay the clean mask on the original image
    overlay = overlay_mask(mask_clean, img)
    
    # Circle the biggest contour on the overlay image
    circled = circle_contour(overlay, big_leaf_contour)
    
    # Display the result with the contour circled
    show(circled)
    
    # Convert the final result back to BGR color space
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    
    return bgr  



#--------------------------------------------------feature extraction----------------------------------------------------------------------------


def color(image): 
    """
    Function to extract color features from an image using a color histogram.

    Parameters:
    image (ndarray): The input image from which color features will be extracted.

    This function calculates the color histogram for the three color channels (B, G, R) 
    and returns the histogram values in 8 bins.
    
    Returns:
    histr (ndarray): Histogram values for each color channel.
    """
    bins = 8  # Number of bins for the histogram (for each color channel)
    
    # Define the color channels (Blue, Green, Red)
    color = ('b', 'g', 'r')  
    
    # Loop over the color channels and calculate histograms for each channel
    for i, col in enumerate(color):  
        histr = cv2.calcHist([image], [i], None, [bins], [0, 256])  # Calculate the histogram for the current color channel
        
        '''
        Uncomment this block if you want to see the histogram plot and values:
        
        var1 = input("Do you want to see histogram plot and color values?:\n")
        if var1 == 'y' or var1 == 'Y':
            plt.plot(histr, color=col)  # Plot the histogram
            plt.xlim([0, bins])  # Set x-axis limits
            plt.xlabel("Total number bins")
            plt.ylabel("Total number of pixels")
            plt.title("HISTOGRAM PLOT")
            plt.show()
            print("NUMBER OF PIXELS IN ALL {} BINS:\n".format(bins)) 
            print(histr)  # Print histogram values
            return histr
        '''
        
        return histr  



def texture(image): 
    """
    Function to extract texture features from an image using Haralick texture descriptors.

    Parameters:
    image (ndarray): The input image from which texture features will be extracted.

    This function converts the image to grayscale and calculates the Haralick texture features using Mahotas.
    
    Returns:
    harlk (ndarray): Array containing 13 Haralick texture features.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    
    # Compute the Haralick texture features
    harlk = mahotas.features.haralick(gray).mean(axis=0)  # Compute the mean of the Haralick feature vectors
    
    '''
    Uncomment this block if you want to see the texture moments:
    
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
    '''
    
    return harlk  


def shape(image): 
    """
    Function to extract shape features from an image using Hu moments.

    Parameters:
    image (ndarray): The input image from which shape features will be extracted.

    This function converts the image to grayscale and calculates the Hu moments, 
    which are invariant to scale, rotation, and translation.
    
    Returns:
    momnt (ndarray): Array of 7 Hu moment values.
    """
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    
    # Compute the Hu moments from the image moments and flatten them into a 1D array
    momnt = cv2.HuMoments(cv2.moments(image)).flatten()  
    
    '''
    Uncomment this block if you want to see the shape moments:
    
    var3 = input("Do you want to see shape moments?:\n")
    if var3 == 'y' or var3 == 'Y':
        print(momnt)
        return momnt
    '''
    
    return momnt  


#--------------------------------------------------------------------------Data Tabulation---------------------------------------------------------------------------

def Data(path):
    """
    Function to generate a DataFrame containing color, texture, and shape features from a set of training images.
    
    Parameters:
    path (list): List of file paths for the training images.
    
    This function reads images from the specified path, extracts color, texture, and shape features, 
    and stores them in a DataFrame along with their labels.
    
    Returns:
    DF (DataFrame): Pandas DataFrame containing feature values for all images.
    label (list): List of labels corresponding to each image.
    """
    
    # Get the list of image file names in the training directory
    Trainpath1 = os.listdir(r"C:\Users\Varsha\Desktop\TrainData")
    
    label = []  # Initialize an empty list for storing labels

    # Iterate over the image file names and assign labels based on the file name
    for file in Trainpath1:
        y = file.split('.')[0]  # Extract the first part of the file name
        if y == 'd':  # If the file name starts with 'd', label it as 1
            label.append(1)
        elif y == 'h':  # If the file name starts with 'h', label it as 0
            label.append(0)

    q = 0  # Initialize a counter for sample numbering
    img = []  # Placeholder for images (not used in the code, but kept for consistency)

    # Create a dictionary to store the feature values and labels
    d = {'Sample': [], 'Label': label}
    
    # Create keys in the dictionary for color features (8 values)
    for i in range(8):
        d['value' + str(i)] = []

    # Create keys in the dictionary for texture features (13 values)
    for j in range(13):
        d['value' + str(8 + j)] = []

    # Create keys in the dictionary for shape features (7 values)
    for k in range(7):
        d['value' + str(21 + k)] = []

    # Iterate over each image path and extract features
    for img in path:
        n = cv2.imread(img)  # Read the image

        # Call the feature extraction functions
        colorCall = color(n)  # Extract color features
        texCall = texture(n)  # Extract texture features
        shapeCall = shape(n)  # Extract shape features

        d['Sample'].append(q)  # Add sample number to the dictionary
        
        # Add the color feature values to the dictionary
        for i in range(8):
            d['value' + str(i)].append(colorCall[i][0])

        # Add the texture feature values to the dictionary
        for i in range(13):
            d['value' + str(8 + i)].append(texCall[i])

        # Add the shape feature values to the dictionary
        for i in range(7):
            d['value' + str(21 + i)].append(shapeCall[i])

        q += 1  # Increment the sample number

    # Convert the dictionary to a Pandas DataFrame
    DF = pd.DataFrame(d)
    
    return DF, label  

def TData():
    """
    Function to generate a DataFrame containing color, texture, and shape features for a single test image.
    
    This function is similar to 'Data', but for a single test image. It extracts color, texture, and shape features,
    and stores them in a DataFrame for later use in testing.
    
    Returns:
    DF (DataFrame): Pandas DataFrame containing feature values for the test image.
    """
    
    q = 0  # Initialize a counter for sample numbering

    # Create a dictionary to store the feature values
    d = {'Sample': []}

    # Create keys in the dictionary for color features (8 values)
    for i in range(8):
        d['value' + str(i)] = []

    # Create keys in the dictionary for texture features (13 values)
    for j in range(13):
        d['value' + str(8 + j)] = []

    # Create keys in the dictionary for shape features (7 values)
    for k in range(7):
        d['value' + str(21 + k)] = []

    # Read the test image
    n = cv2.imread(r"C:\Users\Varsha\Desktop\TestData\test1.png")

    # Call the feature extraction functions
    colorCall = color(n)  # Extract color features
    texCall = texture(n)  # Extract texture features
    shapeCall = shape(n)  # Extract shape features

    d['Sample'].append(q)  # Add sample number to the dictionary

    # Add the color feature values to the dictionary
    for i in range(8):
        d['value' + str(i)].append(colorCall[i][0])

    # Add the texture feature values to the dictionary
    for i in range(13):
        d['value' + str(8 + i)].append(texCall[i])

    # Add the shape feature values to the dictionary
    for i in range(7):
        d['value' + str(21 + i)].append(shapeCall[i])

    q += 1  # Increment the sample number

    # Convert the dictionary to a Pandas DataFrame
    DF = pd.DataFrame(d)
    
    return DF  


#------------------------------------------------------------------------Classification-----------------------------------------------------------------------

def KNN(frame, label, thres): 
    """
    Function to perform K-Nearest Neighbors (KNN) classification on the provided dataset.
    
    Parameters:
    frame (DataFrame): The feature matrix (input data).
    label (list): The labels corresponding to each sample.
    thres (float): The test size ratio (percentage of data to use for testing).
    
    This function splits the data into training and test sets, trains a KNN classifier, 
    and then evaluates it on the test set. The model is also saved as a .sav file using pickle.
    
    Returns:
    ypred (ndarray): Predicted labels for the test data.
    ac (float): Accuracy of the model in percentage.
    """
    # Split the data into training (Xtr, Ytr) and test sets (Xte, Yte) based on the specified threshold
    Xtr, Xte, Ytr, Yte = train_test_split(frame, label, test_size=thres)

    # Create a KNN classifier with 5 neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Train the classifier on the training data
    knn.fit(Xtr, Ytr)
    
    # Save the trained model using pickle
    f = 'knn_model.sav'
    pickle.dump(knn, open(f, 'wb'))

    # Predict labels for the test data
    ypred = knn.predict(Xte)

    # Calculate accuracy of the KNN model on the test set
    ac = metrics.accuracy_score(Yte, ypred) * 100
    
    return ypred, ac  


def SVM(frame, label, thres):
    """
    Function to perform Support Vector Machine (SVM) classification on the provided dataset.
    
    Parameters:
    frame (DataFrame): The feature matrix (input data).
    label (list): The labels corresponding to each sample.
    thres (float): The test size ratio (percentage of data to use for testing).
    
    This function splits the data into training and test sets, trains an SVM classifier, 
    and then evaluates it on the test set.
    
    Returns:
    ypred (ndarray): Predicted labels for the test data.
    ac (float): Accuracy of the model in percentage.
    """
    # Split the data into training (Xtr, Ytr) and test sets (Xte, Yte) based on the specified threshold
    Xtr, Xte, Ytr, Yte = train_test_split(frame, label, test_size=thres)

    # Create an SVM classifier with default parameters
    svm = SVC(gamma='scale')
    
    # Train the classifier on the training data
    svm.fit(Xtr, Ytr)

    # Predict labels for the test data
    ypred = svm.predict(Xte)

    # Calculate accuracy of the SVM model on the test set
    ac = metrics.accuracy_score(Yte, ypred) * 100
    
    return ypred, ac  


def Bay(frame, label, thres):
    """
    Function to perform Gaussian Naive Bayes (Bay) classification on the provided dataset.
    
    Parameters:
    frame (DataFrame): The feature matrix (input data).
    label (list): The labels corresponding to each sample.
    thres (float): The test size ratio (percentage of data to use for testing).
    
    This function splits the data into training and test sets, trains a Naive Bayes classifier, 
    and then evaluates it on the test set.
    
    Returns:
    ypred (ndarray): Predicted labels for the test data.
    ac (float): Accuracy of the model in percentage.
    """
    # Split the data into training (Xtr, Ytr) and test sets (Xte, Yte) based on the specified threshold
    Xtr, Xte, Ytr, Yte = train_test_split(frame, label, test_size=thres)

    # Create a Gaussian Naive Bayes classifier
    gnb = GaussianNB()
    
    # Train the classifier on the training data
    gnb.fit(Xtr, Ytr)

    # Predict labels for the test data
    ypred = gnb.predict(Xte)

    # Calculate accuracy of the Naive Bayes model on the test set
    ac = metrics.accuracy_score(Yte, ypred) * 100
    
    return ypred, ac  



def KNN1(frame1, label1, frame2):
    """
    Function to perform K-Nearest Neighbors (KNN) classification on two separate datasets.
    
    Parameters:
    frame1 (DataFrame): Feature matrix for training data.
    label1 (list): Labels corresponding to the training data.
    frame2 (DataFrame): Feature matrix for test data.
    
    This function trains a KNN model on one dataset (frame1 and label1) and predicts the labels 
    for a separate test dataset (frame2) without splitting the data.
    
    Returns:
    ypred (ndarray): Predicted labels for the test data (frame2).
    """
    # Extract the training data and labels from frame1 and label1
    Xtr, Ytr = (frame1(1), label1[1])

    # Test data (frame2)
    Xte = frame2

    # Create a KNN classifier with 5 neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Train the classifier on the training data
    knn.fit(Xtr, Ytr)

    # Predict labels for the test data
    ypred = knn.predict(Xte)

    # Print the predicted labels for the test data
    print(ypred)

    return ypred  


#-----------------------------------------------------------------------------------Main---------------------------------------------------------------------------


# Load training data, extract features, and save the features into CSV files
traindata, label = Data(Trainpath)
traindata.to_csv(r"C:\Users\Varsha\Desktop\TrainData.csv")  # Save training data features to CSV
testdata = TData()
testdata.to_csv(r"C:\Users\Varsha\Desktop\TestData.csv")  # Save test data features to CSV

# Run KNN on the training data and test it on the test data
knnt = KNN1(traindata, label, testdata)

# Start performing KNN with varying test/train split ratios
thres = 0.1
while thres < 0.4:
    thres += 0.1  # Increment the test size by 0.1 each iteration
    
    # Perform KNN with the current test size
    knntr, ack = KNN(traindata, label, thres)
    knnv = np.array(knntr)  # Convert predictions to numpy array
    ackall = np.append(ack)  # Append accuracy for tracking
    
    # Print accuracy
    print(ackall)
    
    # Perform SVM with the current test size
    svmtr, acs = SVM(traindata, label, thres)
    svmv = np.array(svmtr)  # Convert predictions to numpy array
    
    # Perform Naive Bayes with the current test size
    baytr, acb = Bay(traindata, label, thres)
    bayv = np.array(baytr)  # Convert predictions to numpy array

# Print the algorithm names and test sizes along with their accuracies
print("Algorithm        " + "Test Size       " + "Accuracy")

# Perform KNN for different test sizes
knntr, ack1 = KNN(traindata, label, 0.2)
knntr, ack2 = KNN(traindata, label, 0.25) 
knntr, ack3 = KNN(traindata, label, 0.3) 
knntr, ack4 = KNN(traindata, label, 0.5)
knntr, ack5 = KNN(traindata, label, 0.6)

# Collect all KNN accuracies
ackall1 = np.append(ack1, ack2)
ackall2 = np.append(ackall1, ack3)
ackall3 = np.append(ackall2, ack4)
ackall = np.append(ackall3, ack5)

# Print results for KNN
print("Total number of dataset - 30 images: 15 Healthy leaf images and 15 Diseased leaf images")
print("\n")
print("Algorithm        " + "Train/Test Size      " + "No. of Images  " + "        Accuracy")

print("   KNN           " + "   80/20                  " + str(int(0.80 * 30)) + ":" + str(int(0.20 * 30)) + "\t\t     " + str(ackall[0]))
print("                 " + "   75/25                  " + str(int(0.75 * 30)) + ":" + str(int(0.25 * 30)) + "\t\t     " + str(ackall[1]))
print("                 " + "   70/30                  " + str(int(0.70 * 30)) + ":" + str(int(0.30 * 30)) + "\t\t     " + str(ackall[2]))
print("                 " + "   50/50                  " + str(int(0.50 * 30)) + ":" + str(int(0.50 * 30)) + "\t     " + str(ackall[3]))
print("                 " + "   40/60                  " + str(int(0.40 * 30)) + ":" + str(int(0.60 * 30)) + "\t     " + str(ackall[4]))

# Perform SVM for different test sizes
svmtr, acs1 = SVM(traindata, label, 0.2)
svmtr, acs2 = SVM(traindata, label, 0.25)
svmtr, acs3 = SVM(traindata, label, 0.3)
svmtr, acs4 = SVM(traindata, label, 0.5)
svmtr, acs5 = SVM(traindata, label, 0.6)

# Collect all SVM accuracies
acsall1 = np.append(acs1, acs2)
acsall2 = np.append(acsall1, acs3)
acsall3 = np.append(acsall2, acs4)
acsall = np.append(acsall3, acs5)

# Print results for SVM
print("\n")
print("   SVM           " + "   80/20                 " + str(int(0.80 * 30)) + ":" + str(int(0.20 * 30)) + "\t\t     " + str(acsall[0]))
print("                 " + "   75/25                 " + str(int(0.75 * 30)) + ":" + str(int(0.25 * 30)) + "\t\t     " + str(acsall[1]))
print("                 " + "   70/30                 " + str(int(0.70 * 30)) + ":" + str(int(0.30 * 30)) + "\t\t     " + str(acsall[2]))
print("                 " + "   50/50                 " + str(int(0.50 * 30)) + ":" + str(int(0.50 * 30)) + "\t\t     " + str(acsall[3]))
print("                 " + "   40/60                 " + str(int(0.40 * 30)) + ":" + str(int(0.60 * 30)) + "\t\t     " + str(acsall[4]))

# Perform Naive Bayes for different test sizes
baytr, acb1 = Bay(traindata, label, 0.2)
baytr, acb2 = Bay(traindata, label, 0.25)
baytr, acb3 = Bay(traindata, label, 0.3)
baytr, acb4 = Bay(traindata, label, 0.5)
baytr, acb5 = Bay(traindata, label, 0.6)

# Collect all Naive Bayes accuracies
acball1 = np.append(acb1, acb2)
acball2 = np.append(acball1, acb3)
acball3 = np.append(acball2, acb4)
acball = np.append(acball3, acb5)

# Print results for Naive Bayes
print("\n")
print("NAIVE BAYES      " + "   80/20                " + str(int(0.80 * 30)) + ":" + str(int(0.20 * 30)) + "\t\t     " + str(acball[0]))
print("                 " + "   75/25                " + str(int(0.75 * 30)) + ":" + str(int(0.25 * 30)) + "\t\t     " + str(acball[1]))
print("                 " + "   70/30                " + str(int(0.70 * 30)) + ":" + str(int(0.30 * 30)) + "\t\t     " + str(acball[2]))
print("                 " + "   50/50                " + str(int(0.50 * 30)) + ":" + str(int(0.50 * 30)) + "\t\t     " + str(acball[3]))
print("                 " + "   40/60                " + str(int(0.40 * 30)) + ":" + str(int(0.60 * 30)) + "\t\t     " + str(acball[4]))

#%%  

# Function to test a single image using SVM and KNN classifiers
def test_one():
    img = np.array(Image.open(r"C:\Users\Varsha\Desktop\Trainimages\Test\test3.jpg"))  # Load test image as array

    X = []  # Placeholder for feature vectors
    Y = []  # Placeholder for labels

    # Load healthy leaf data
    HFiles = glob.glob("HData/*.png")
    for file in HFiles:
        X.append(np.array(Image.open(file).convert("L").resize((720, 720), Image.ANTIALIAS)).flatten())    
        Y.append(0)  # Healthy leaves labeled as 0
    
    # Load diseased leaf data
    DFiles = glob.glob("DData/*.png")
    for file in DFiles:
        X.append(np.array(Image.open(file).convert("L").resize((720, 720), Image.ANTIALIAS)).flatten())    
        Y.append(1)  # Diseased leaves labeled as 1
    
    X = np.array(X)  # Convert feature vectors to array
    Y = np.array(Y).reshape(-1, 1)  # Reshape labels to match the model input

    # Train an SVM classifier
    a_svm = SVC()
    a_svm.fit(X, Y)

    # Predict the class of the test image using the trained SVM model
    a_svm = a_svm.predict(img)[0]
    
    # Print the predicted class for the test image
    print(a_svm)
