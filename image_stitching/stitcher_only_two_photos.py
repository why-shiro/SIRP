import cv2 
import os
import matplotlib.pyplot as plt
import numpy as np

# Convert .tif images to .png and save to /outputs
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Load the images
img1 = cv2.imread('./image_stitching/outputs/4.png')
img2 = cv2.imread('./image_stitching/outputs/3.png')

# ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)  # Match descriptors

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

print("Number of matches:", len(matches))

# Move same parts of the images to the same coordinates
# Calculate Homography
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp the first image to the second image
h, w = img2.shape[:2]
warped_img = cv2.warpPerspective(img1, M, (w, h + img1.shape[0]))

# Combine the images with weighted addition
alpha = 0.5
beta = 1.0 - alpha
blended = cv2.addWeighted(warped_img[:h, :w], alpha, img2, beta, 0)

# Place the blended part into the result
warped_img[:h, :w] = blended
result = warped_img

cv2.imwrite('./result.png', result)

# Show the result
plt.figure(figsize=(20, 10))
plt.imshow(result)
plt.axis('off')
plt.show()