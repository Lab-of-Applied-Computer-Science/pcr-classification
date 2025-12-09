import cv2
import numpy as np
import os

# Load the binary mask image
input_image_path = "./breast-cancer/Masks/f03f7a72-707a-45fe-a0a8-ae23816d12ec_0_f03f7a72-707a-45fe-a0a8-ae23816d12ec.png"
output_folder = "./breast-cancer/Masks"
# Extract the original file name without extension
base_filename = os.path.splitext(os.path.basename(input_image_path))[0]

# Load the image in grayscale
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Find contours (which represent the white areas in the binary mask)
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over each contour and create a separate image for each one
for i, contour in enumerate(contours):
    # Create a black canvas the same size as the input image
    mask = np.zeros_like(image)

    # Draw the current contour on the mask in white
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Generate the output image path with the modified name
    output_image_path = os.path.join(output_folder, f"{base_filename}-idx{i+1}.png")

    # Save the mask to a new image file
    cv2.imwrite(output_image_path, mask)

    print(f"Saved: {output_image_path}")
