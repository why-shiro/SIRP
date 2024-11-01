import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Debug mode
debug = False

# List of image file paths in reverse order
image_files = ['./image_stitching/outputs/4.png', './image_stitching/outputs/3.png', './image_stitching/outputs/2.png', './image_stitching/outputs/1.png', './image_stitching/outputs/0.png']

# Load the first image as the starting result
result = cv2.imread(image_files[0])

# ORB detector
orb = cv2.ORB_create()

for i in range(1, len(image_files)):
    # Load the next image to merge
    next_img = cv2.imread(image_files[i])

    # Detect keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(result, None)
    kp2, des2 = orb.detectAndCompute(next_img, None)

    # Create BFMatcher object and find matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Number of matches between images {i-1} and {i}:", len(matches))

    # Calculate Homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        print("Homography could not be computed for images", i-1, "and", i)
        continue
    else:
        print(f"Homography matrix for images {i-1} and {i}:")
        print(M)

    # Warp the current result to align with the next image
    h, w = next_img.shape[:2]
    warped_img = cv2.warpPerspective(result, M, (w, h + result.shape[0]))

    # Adjusted weights for blending
    alpha = 0.0  # Higher weight for the warped image to retain brightness
    beta = 1.0 - alpha  # Complementary weight for the second image

    # Combine the images with adjusted weights
    blended = cv2.addWeighted(warped_img[:h, :w], alpha, next_img, beta, 0)

    # Place the blended part into the result
    warped_img[:h, :w] = blended
    result = warped_img  # Update result for next iteration

    # Optionally, draw intersection lines if debug mode is on
    if debug:
        height, width = result.shape[:2]
        corners = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, M)
        for j in range(4):
            start_pt = tuple(map(int, transformed_corners[j][0]))
            end_pt = tuple(map(int, transformed_corners[(j + 1) % 4][0]))
            cv2.line(result, start_pt, end_pt, (0, 255, 0), 3)  # Draw green lines

# Save the final stitched result
cv2.imwrite('./final_result.png', result)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Final Stitched Image')
plt.show()
