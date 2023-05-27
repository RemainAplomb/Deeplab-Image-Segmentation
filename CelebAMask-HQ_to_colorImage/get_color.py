import os
from PIL import Image
import glob
import numpy as np
from utils import make_folder


# Face, Hair, Body, Accessories, Background
# 0 - Background
# 1 - Face
# 2 - Hair
# 3 - Body
# 4 - Accesories
color_list = [[0, 0, 0], [1, 1, 1], [1, 1, 1], [4, 4, 4], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2], [4, 4, 4], [4, 4, 4], [4, 4, 4], [3, 3, 3], [3, 3, 3]]


# Face, Hair, Body, Background
# 0 - Background
# 1 - Face
# 2 - Hair
# 3 - Body
# color_list = [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3], [3, 3, 3]]



folder_base = '/content/CelebAMask-HQ-mask'
folder_save = '/content/CelebAMask-HQ-mask-color'
img_num = 30000

make_folder(folder_save)

for k in range(img_num):
    filename = os.path.join(folder_base, str(k) + '.png')
    if (os.path.exists(filename)):
        im_base = np.zeros((512, 512, 3))
        im = Image.open(filename)
        im = np.array(im)
        for idx, color in enumerate(color_list):
            im_base[im == idx] = color
    filename_save = os.path.join(folder_save, str(k) + '.png')
    result = Image.fromarray((im_base).astype(np.uint8))
    print (filename_save)
    result.save(filename_save)
