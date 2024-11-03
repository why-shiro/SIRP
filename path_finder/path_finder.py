import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

image = cv2.imread('./path_finder/photos/example.png')

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
red_points = []
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        red_points.append((cx, cy))

print("Kırmızı noktalar:", red_points)

image[red_mask > 0] = [255, 255, 255]

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

if len(red_points) >= 2:
    start = red_points[0]
    end = red_points[1]
else:
    print("Yeterli kırmızı nokta bulunamadı.")
    exit()

graph = nx.Graph()

height, width = binary_image.shape
for y in range(height):
    for x in range(width):
        if binary_image[y, x] == 255:
            graph.add_node((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-komşuluk
                neighbor_x, neighbor_y = x + dx, y + dy
                if 0 <= neighbor_x < width and 0 <= neighbor_y < height and binary_image[neighbor_y, neighbor_x] == 255:
                    graph.add_edge((x, y), (neighbor_x, neighbor_y))

try:
    shortest_path = nx.shortest_path(graph, source=start, target=end, method='dijkstra')
    print("En kısa yol bulundu:", shortest_path)
except nx.NetworkXNoPath:
    print("Başlangıç ve bitiş noktaları arasında yol yok!")
    shortest_path = []

result_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
for i in range(len(shortest_path) - 1):
    cv2.line(result_image, shortest_path[i], shortest_path[i + 1], (0, 0, 255), thickness=3)  # Kalınlık = 3

# Sonucu göster
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Shortest Path")
plt.axis("off")
plt.show()
