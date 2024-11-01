import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img

directory = "./image_stitching/outputs/"  # Current directory
img_files = [img for img in os.listdir(directory) if img.endswith('.png')]

print(f"Found {len(img_files)} images in the directory.")

# Read images and filter out None values
img_list = []
for img in sorted(img_files):
    img_path = os.path.join(directory, img)
    image = cv2.imread(img_path)
    if image is not None:
        img_list.append(image)

orb = cv2.ORB_create(nfeatures=2000)

total_images = len(img_list)
stitch_count = 0  # Count of stitched images

while len(img_list) > 1:
    img1 = img_list.pop(0)
    img2 = img_list.pop(0)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good = [m for m, n in matches if m.distance < 0.6 * n.distance]

    MIN_MATCH_COUNT = 5

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        result = warpImages(img2, img1, M)
        img_list.insert(0, result)
        
        stitch_count += 1  # Increment the stitch count

        # Calculate and print the progress
        progress = (stitch_count / (total_images - 1)) * 100  # -1 because we start with two images
        print(f"Stitching progress: {progress:.2f}%")

cv2.imwrite('result.jpg', result)  # Save in BGR format
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
plt.imshow(result_rgb)
plt.axis('off')  # Hide axes
plt.show()
