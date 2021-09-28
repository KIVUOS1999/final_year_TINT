import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib
from PIL import Image 
import PIL 

data = 'images/'
arr = []

for j in tqdm(os.listdir(data)):
    sub_dir = data + j
    a = cv2.imread(sub_dir, 0)
    arr.append(a)

                #a = Image.fromarray(a)
                #a.save('images/'+i.split('.')[0]+'.png')

X = np.array(arr)
print(X.shape)

np.save("numpy_saves/face64X64_filtered", X)