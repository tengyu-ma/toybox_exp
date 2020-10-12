import conf
import glob
import torch
import imageio


class ToyboxSquareData(torch.utils.data.Dataset):
    def __init__(self, root, dataset, ratio, transform=None):
        self.root = root
        self.dir = f'{self.root}/*/{dataset}/*/*/{ratio}.png'
        self.files = sorted(glob.glob(self.dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        ca, no, tr, fr, ratio = self._get_img_info(file)

        img = imageio.imread(file)

        label = conf.ALL_CA.index(ca)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, file

    @staticmethod
    def _get_img_info(img_path):
        img_path, _ = img_path.split('.')
        info = img_path.split('/')
        ca = info[-5]
        no = info[-3]
        tr, fr = info[-2].split('_')
        ratio = info[-1]
        return ca, int(no), tr, int(fr), int(ratio)