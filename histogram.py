# Import necessary modules 
import cv2
from matplotlib import pyplot as plt

# Read two images for comparison (Healthy and Diseased)
image1 = cv2.imread(r"C:\Users\Varsha\Desktop\TrainData\h.8.png")  # Load healthy leaf image
image2 = cv2.imread(r"C:\Users\Varsha\Desktop\TrainData\d.5.png")  # Load diseased leaf image

# Define the color channels (Blue, Green, Red)
color = ('b', 'g', 'r')

# Set the number of bins for the histogram
bins = 8 

# Loop through the color channels to calculate histograms for both images
for i, col in enumerate(color):
    # Calculate histogram for the current color channel of the first image
    histr1 = cv2.calcHist([image1], [i], None, [bins], [0, 256])
    
    # Calculate histogram for the current color channel of the second image
    histr2 = cv2.calcHist([image2], [i], None, [bins], [0, 256])

# Create a scatter plot to compare the histograms of both images
plt.scatter(histr1, histr2)

# Show the plot
plt.show()
