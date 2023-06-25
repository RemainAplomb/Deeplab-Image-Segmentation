import os
from PIL import Image
import glob
import numpy as np
from utils import make_folder
# 1 - Face
# 2 - Hair
# 3 - Body
# 4 - Accesories

# 0 - Background
# 1 - Hair
# 2 - Face
# 3 - Top-Body-Skin
# 4 - Top-Body-Clothes
# 5->3 - Lower-Body-Skin
# 6->4 - Lower-Body-Clothes
# 7->5 - Upper-Accessories
# 8->5 - Lower-Accessories

# 0 - Background
# 1 - Hair
# 2 - Face
# 3 - Skin
# 4 - Clothes
# 5 - Accesories
# label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
#               'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
color_list = [[0, 0, 0], [2, 2, 2], [2, 2, 2], [5, 5, 5], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1], [5, 5, 5], [5, 5, 5], [5, 5, 5], [3, 3, 3], [4, 4, 4]]

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
