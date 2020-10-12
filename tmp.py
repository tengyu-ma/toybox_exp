import os
import conf
import imageio
import numpy as np
import matplotlib.pyplot as plt


file = os.path.join('/media/tengyu/DataU/Data/Toybox/frame6_1920_1080/truck_26/truck_26_pivothead_rzplus.mp4_091.jpeg')

img = imageio.imread(file)
plt.imshow(img)
plt.show()
print(img.shape)

top = 1020
height = 1080 - top

left = 1025
right = 1230

new_img = img[top: top+height, left:right]
# #
# color = (new_img[0, 0].astype(np.float) + new_img[0, -1].astype(np.float)) // 2
# color = color.astype(np.uint8)
# color = np.expand_dims(color, axis=0)
# color = np.expand_dims(color, axis=0)
# padding = np.repeat(color, 73, axis=0)
# padding = np.repeat(padding, right - left, axis=1)

# new_img = np.vstack([new_img, padding])
# #
plt.imshow(new_img)
plt.show()

print(f'left {left}')
print(f'top {top}')
print(f'width {right - left}')
print(f'height {height}')
print(new_img.shape)
#
#
#
# imageio.imsave(file.replace('frame6_1920_1080', 'cropped_square'), new_img)


"""
car 24 rzminus 49
cat 08 rzminus 49
cat 18 rzplus 85
cat 18 rzplus 103
cup 22 rzplus 73
spoon 07 rzplus 37
spoon 11 rzminus 37, rzminus 97
spoon 24 rzplus 55, rzplus 79
spoon 28 rzminus 37
truck 26 rzplus 91
"""


import os
import shutil
from glob import glob

files = glob(f'{conf.ToyboxSquaresRz}/*/*/*/*/*.png')
files = sorted(files)

file_check = {}

for file in files:
    base = '/'.join(file.split('/')[:10])
    img = '/'.join(file.split('/')[10:])
    file_check[base] = file_check.get(base, [])
    file_check[base].append(img)

for k, v in file_check.items():
    if len(v) != 144:
        print(k, len(v), v)

from glob import glob
folders = glob(f'{conf.ToyboxSquares}/*/*/*/*')

res = {}
for folder in folders:
    base, fr = folder.split('_')
    fr = int(fr)
    if base.endswith('rzplus') or base.endswith('rzminus'):
        res[base] = max(res.get(base, 0), fr)

# for k, v in res.items():
#     if v > 145:
#         print(k)
from collections import Counter
print(sorted(set(res.values())))
fr_cases = sorted(set(res.values()))

keep_fr = {}
for fr_case in fr_cases:
    all_fr = list(range(1, fr_case + 1, 6))
    num_removed = len(all_fr) - 18
    if num_removed == 0:
        keep_fr[fr_case] = all_fr
    else:
        removed = list(np.linspace(1, fr_case, num_removed + 2))[1: -1]
        for removed_fr_indicator in removed:
            removed_fr = min(all_fr, key=lambda x: abs(x - removed_fr_indicator))
            all_fr.remove(removed_fr)
        keep_fr[fr_case] = all_fr

files = glob(f'{conf.ToyboxSquares}/*/*/*/*/*.png')

for src_root, case in res.items():
    files = glob(f'{src_root}_*/*.png')
    keeped_fr = keep_fr[case]
    for file in files:
        ca = file.split('/')[7]
        no = int(file.split('/')[9])
        tr, fr = file.split('/')[10].split('_')
        fr = int(fr)
        # if not (ca == 'car' and no == 24 and tr == 'rzminus' and fr == 49) \
        #     and not (ca == 'cat' and no == 8 and tr == 'rzminus' and fr == 49) \
        #     and not (ca == 'cat' and no == 18 and tr == 'rzplus' and fr == 85) \
        #     and not (ca == 'cat' and no == 18 and tr == 'rzplus' and fr == 103) \
        #     and not (ca == 'cup' and no == 22 and tr == 'rzplus' and fr == 73) \
        #     and not (ca == 'spoon' and no == 7 and tr == 'rzplus' and fr == 37) \
        #     and not (ca == 'spoon' and no == 11 and tr == 'rzminus' and fr == 37) \
        #     and not (ca == 'spoon' and no == 11 and tr == 'rzminus' and fr == 97) \
        #     and not (ca == 'spoon' and no == 24 and tr == 'rzplus' and fr == 55) \
        #     and not (ca == 'spoon' and no == 24 and tr == 'rzplus' and fr == 79) \
        #     and not (ca == 'spoon' and no == 28 and tr == 'rzminus' and fr == 37) \
        if not (ca == 'truck' and no == 26 and tr == 'rzplus' and fr == 91):
            continue

        fr = int(file.split('/')[-2].split('_')[-1])
        if fr in keeped_fr:
            src = file
            dst = src.replace('squares', 'squares_rz')

            file_name = dst.split('/')[-1]
            zoom, ext = file_name.split('.')
            file_name = f'{int(zoom):03d}.{ext}'

            dst_folder = '/'.join(dst.split('/')[:-1])
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)

            dst = f'{dst_folder}/{file_name}'
            shutil.copy(src, dst)

    print('Done')

