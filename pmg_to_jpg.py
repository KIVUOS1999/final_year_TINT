import cv2
import os
import numpy as np
from tqdm import tqdm

data = 'CroppedYale/'
arr = []

for i in tqdm(os.listdir(data)):
    sub_dir = data + i
    for i in os.listdir(sub_dir):
        if '.' in i:
            if i.split('.')[1] == 'pgm':
                a = cv2.imread(sub_dir +'/'+ i, 0)
                a = cv2.resize(a, (64, 64))
                arr.append(a)

X = np.array(arr)
print(X.shape)

np.save("numpy_saves/face64X64", X)