import os
import conf
import glob
import torch
import imageio
import pandas as pd

from PIL import Image


class ToyboxRawData(torch.utils.data.Dataset):
    def __init__(self, root, ca, no, tr, fr, transform=None):
        self.files = sorted(glob.glob(root), key=self._get_img_info)

        self.df = self._get_df()
        self.df = self._filter(ca, no, tr, fr)
        self.transform = transform

    def squares_ready(self):
        """ Make the cropped_square Toybox ready for different squares levels
        """
        ratios = [1, 0.75, 0.50, 0.25]
        for i, src in enumerate(self.files):
            print(f'{i+1}/{len(self.files)}')
            ca, no, tr, fr = self._get_img_info(src)
            train_or_test = 'test' if no in conf.TEST_NO[ca] else 'train'
            dst_root = f'{conf.ToyboxSquares}/{ca}/{train_or_test}/{no:02d}/{tr}_{fr:03d}'
            if not os.path.exists(dst_root):
                os.makedirs(dst_root)

            img = imageio.imread(src)
            h, w, c = img.shape

            for ratio in ratios:
                h_new = max(int(h * ratio), 1)
                w_new = max(int(w * ratio), 1)

                start_h = int((h - h_new) // 2)
                start_w = int((w - w_new) // 2)
                img_new = img[start_h: start_h + h_new, start_w: start_w + w_new, :]
                img_new = Image.fromarray(img_new).resize((128, 128))
                img_new.save(os.path.join(dst_root, f'{int(ratio * 100):3d}.png'))

    def _filter(self, ca, no, tr, fr):
        df = self.df
        df = df[df['ca'].isin(ca)] if ca else df
        df = df[df['no'].isin(no)] if no else df
        df = df[df['tr'].isin(tr)] if tr else df
        df = df[df['fr'].isin(fr)] if fr else df
        return df.reset_index().drop('index', axis=1)

    def _get_df(self):
        df = pd.DataFrame(self.files, columns=['path'])
        df['ca'] = df['path'].apply(lambda x: self._get_img_info(x)[0])
        df['no'] = df['path'].apply(lambda x: self._get_img_info(x)[1])
        df['tr'] = df['path'].apply(lambda x: self._get_img_info(x)[2])
        df['fr'] = df['path'].apply(lambda x: self._get_img_info(x)[3])
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        info = self.df.loc[index]
        img = imageio.imread(info.path)
        label = conf.ALL_CA.index(info.ca)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, info.path

    @staticmethod
    def _get_img_info(img_path):
        basic, video, _ = os.path.basename(img_path).split('.')
        ca, no, _, tr = basic.split('_')  # category, number, _, transformation
        _, fr = video.split('_')  # _. fra
        return ca, int(no), tr, int(fr)
