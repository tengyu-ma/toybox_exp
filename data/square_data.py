import conf
import glob
import torch
import imageio
import numpy as np


class ToyboxSquareData(torch.utils.data.Dataset):
    def __init__(self, root, dataset, ratio, transform=None):
        self.root = root
        self.dir = f'{self.root}/*/{dataset}/*/*/{ratio}.png'
        print(self.dir)
        self.files = sorted(glob.glob(self.dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        # # ToDo: the following is a temporary code for proto-typing. Will need to well code in the future
        # ratios = ['025.png', '050.png', '075.png', '100.png']
        # tiled_files = [file.replace('100.png', ratio) for ratio in ratios]

        ca, no, tr, fr, ratio = self._get_img_info(file)

        # img_025_050 = np.hstack([imageio.imread(file) for file in tiled_files[:2]])
        # img_050_100 = np.hstack([imageio.imread(file) for file in tiled_files[2:4]])
        #
        # img = np.vstack([img_025_050, img_050_100])
        # import matplotlib.pyplot as plt
        #
        # plt.imshow(img)
        img = imageio.imread(file)

        label = conf.ALL_CA.index(ca)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, file
        # return img, label, '>'.join(tiled_files)

    @staticmethod
    def _get_img_info(img_path):
        img_path, _ = img_path.split('.')
        info = img_path.split('/')
        ca = info[-5]
        no = info[-3]
        tr, fr = info[-2].split('_')
        ratio = info[-1]
        return ca, int(no), tr, int(fr), int(ratio)

