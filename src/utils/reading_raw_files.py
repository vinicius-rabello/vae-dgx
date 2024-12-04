import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import shutil

paths = os.listdir('data/raw')
paths = ['data/raw/' + path for path in paths]

dataset = []
for path in paths:
    with open(path, 'rb') as f:
        unshaped_voxel = np.fromfile(f, dtype=np.uint8)
    images = unshaped_voxel.reshape(1000, 1000, 1000)
    dataset.extend(images)

def crop_image(image):
    crops = []
    for i in range(0, 512, 256):
        for j in range(0, 512, 256):
            crops.append(image[i:i+256, j:j+256])
    return crops

cropped_dataset = []
for image in tqdm(dataset):
    image = cv2.resize(image, (512, 512))
    cropped_dataset.extend(crop_image(image))

if os.path.exists('data/dataset'):
        shutil.rmtree('data/dataset')
os.makedirs('data/dataset/train_set/0')
os.makedirs('data/dataset/test_set/0')

np.random.shuffle(cropped_dataset)
train_set = cropped_dataset[:35200]
test_set = cropped_dataset[35200:]

print('creating training set...')    
for i, image in enumerate(train_set):
    print(f'saving {i}.png to data/dataset/train_set/0...')
    cv2.imwrite(f'data/dataset/train_set/0/{i}.png', 255*image)
    
print('creating test set...')
for i, image in enumerate(test_set):
    print(f'saving {i + 35200}.png to data/dataset/test_set/0...')
    cv2.imwrite(f'data/dataset/test_set/0/{i + 35200}.png', 255*image)
    
print('dataset is ready!')