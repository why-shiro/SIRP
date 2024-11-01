import cv2
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def convert_tif_to_png(images):
    for i in range(len(images)):
        img = cv2.imread('output/' + str(i) + '.png')
        img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
        cv2.imwrite('output/converter/' + str(i) + '.png', img)


images = load_images_from_folder('photos/')
convert_tif_to_png(images)