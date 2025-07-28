Flooded Area Segmentation with Attention U-Net
This project demonstrates how to use a trained Attention U-Net model for automated segmentation of flooded areas in satellite or aerial images.

Code Overview
The provided script performs the following steps:

Load Test Images

Reads images from a specified directory.
Converts images from BGR to RGB color space.
Resizes images to 512x512 pixels.
Normalizes pixel values to the range [0, 1].
Prepares the images as a NumPy array for model input.
Predict Flood Masks

Uses a pre-trained Attention U-Net model to predict segmentation masks for the test images.
Visualize Results

Displays each test image alongside its predicted segmentation mask using Matplotlib for easy comparison.
Code Usage
Make sure you have the following dependencies installed:

*OpenCV (cv2)
*NumPy
*Matplotlib
*TensorFlow or Keras (for model loading and prediction)
Update the image path in the code if your images are located elsewhere.

Python
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Step 1: Load test images
image_paths = sorted(glob('/path/to/your/images/*.jpg'))
img_test = []
for path in image_paths[:3]:  # Load only 3 for example
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to load {path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype('float32') / 255.0
    img_test.append(img)
img_test = np.array(img_test)

# Step 2: Predict using model (ensure 'model' is loaded)
pred = model.predict(img_test)

# Step 3: Display results
for i in range(len(img_test)):
    img = img_test[i]
    pred_img = pred[i]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Image {i+1}")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(pred_img), cmap='gray')
    plt.title(f"Prediction {i+1}")
    plt.axis('off')
    plt.tight_layout()
```
#**Notes**

Replace /path/to/your/images/*.jpg with the actual path to your image files.
Ensure your model variable (model) is defined and loaded with the trained Attention U-Net weights.
Adjust the number of images to load and display as needed.
Repository
For more information on the dataset, model architecture, training, and evaluation, see the rest of this repository.
