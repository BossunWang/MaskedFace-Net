import json
import os
import cv2
import numpy as np


def face_crop(image_path, rect, save_path):
    image = cv2.imread(image_path)  
    
    for i in range(len(rect)):
        rect[i] /= 1024.0
        rect[i] = int(rect[i] * image.shape[0])
    print(rect)
    roi_image = np.copy(image[rect[1]:rect[3], rect[0]:rect[2]])
    print(roi_image.shape)

    box_w = rect[2] - rect[0]
    box_h = rect[3] - rect[1]
    if box_h < box_w:
        padding_size = abs(box_w - box_h) // 2
        padding = cv2.copyMakeBorder(roi_image, padding_size, padding_size, 0, 0, cv2.BORDER_CONSTANT,
                                        value=[0, 0, 0])
    else:
        padding_size = abs(box_w - box_h) // 2
        padding = cv2.copyMakeBorder(roi_image, 0, 0, padding_size, padding_size, cv2.BORDER_CONSTANT,
                                        value=[0, 0, 0])
    
    print(padding.shape)
    new_image = cv2.resize(padding, (112, 112), interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_path, new_image)
    cv2.imshow("face", new_image)
    cv2.waitKey(1)


f = open('../face_dataset/ffhq-dataset-v2.json')
data = json.load(f)

# print(len(data))
# print(data['0']['image']['file_path'])
# print(data['0']['in_the_wild']['face_rect'])

unmasked_data_folder = '../face_dataset/thumbnails128x128'
CMFD_data_folder = '../face_dataset/MaskedFace-Net/CMFD/'
IMFD_data_folder = '../face_dataset/MaskedFace-Net/IMFD/'
target_folder = '../face_dataset/FFHQ_masked_unmasked'

CMFD_type = '_Mask.jpg'
IMFD_type1 = '_Mask_Nose_Mouth.jpg'
IMFD_type2 = '_Mask_Mouth_Chin.jpg'
IMFD_type3 = '_Mask_Chin.jpg'

for i in range(len(data)):
    image_path = data[str(i)]['image']['file_path']
    face_landmarks = data[str(i)]['image']['face_landmarks']

    folder = image_path.split('/')[0]
    id = image_path.split('/')[1]
    file_name = image_path.split('/')[-1]

    unmasked_image_path = image_path.replace(folder, unmasked_data_folder)
    CMFD_image_path = image_path.replace(folder, CMFD_data_folder + id)
    IMFD_image_path = image_path.replace(folder, IMFD_data_folder + id)
    
    CMFD_image_path = CMFD_image_path.replace('.png', CMFD_type)
    IMFD_image_path1 = IMFD_image_path.replace('.png', IMFD_type1)
    IMFD_image_path2 = IMFD_image_path.replace('.png', IMFD_type2)
    IMFD_image_path3 = IMFD_image_path.replace('.png', IMFD_type3)

    print(unmasked_image_path)
    print(CMFD_image_path)
    print(IMFD_image_path1)
    print(IMFD_image_path2)
    print(IMFD_image_path3)

    target_data_folder = os.path.join(target_folder, file_name.split('.png')[0])

    if not os.path.isdir(target_data_folder):
        os.mkdir(target_data_folder)

    unmasked_target_image_path = os.path.join(target_data_folder, unmasked_image_path.split('/')[-1]).replace('.png', '.jpg')
    CMFD_target_image_path = os.path.join(target_data_folder, CMFD_image_path.split('/')[-1])
    IMFD_target_image_path1 = os.path.join(target_data_folder, IMFD_image_path1.split('/')[-1])
    IMFD_target_image_path2 = os.path.join(target_data_folder, IMFD_image_path2.split('/')[-1])
    IMFD_target_image_path3 = os.path.join(target_data_folder, IMFD_image_path3.split('/')[-1])

    print(unmasked_target_image_path)
    print(CMFD_target_image_path)
    print(IMFD_target_image_path1)
    print(IMFD_target_image_path2)
    print(IMFD_target_image_path3)

    min_x, min_y, max_x, max_y = 1024, 1024, 0, 0
    for landmark in face_landmarks:
        if min_x > landmark[0] and landmark[0] > 0:
            min_x = landmark[0]
        if min_y > landmark[1] and landmark[1] > 0:
            min_y = landmark[1]
        if max_x < landmark[0]:
            max_x = landmark[0]
        if max_y < landmark[1]:
            max_y = landmark[1]

    if min_x > max_x or min_y > max_x:
        continue

    rect = [min_x, (max_y - min_y) // 2, max_x, max_y]
    # if os.path.isfile(unmasked_image_path):        
    #     face_crop(unmasked_image_path, rect.copy(), unmasked_target_image_path)
    if os.path.isfile(CMFD_image_path):        
        face_crop(CMFD_image_path, rect.copy(), CMFD_target_image_path)
    # if os.path.isfile(IMFD_image_path1):        
    #     face_crop(IMFD_image_path1, rect.copy(), IMFD_target_image_path1)
    # if os.path.isfile(IMFD_image_path2):        
    #     face_crop(IMFD_image_path2, rect.copy(), IMFD_target_image_path2)
    # if os.path.isfile(IMFD_image_path3):        
    #     face_crop(IMFD_image_path3, rect.copy(), IMFD_target_image_path3)
    