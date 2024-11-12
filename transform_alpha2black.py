import cv2
import numpy as np
import os

def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image.shape[2] == 4: 
                alpha_channel = image[:, :, 3]
                image[alpha_channel == 0] = [0, 0, 0, 255]  
            
            cv2.imwrite(image_path, image)

process_images_in_folder('./data/single_building_images/')