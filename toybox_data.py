import conf
import math
import torch
import imageio
import numpy as np
import pandas as pd

from glob import glob


class ToyboxData(torch.utils.data.Dataset):
    def __init__(self, root, tr, nview, ratio, mode, dataset, transform=None):
        """ The Toybox dataset designed for squares_same_nview processed data

        Args:
            root: str,
                The path where you stored the squares_same_nview Toybox data
            tr: List[str], 'rzplus', 'rzminus', 'ryplus', 'ryminus', 'rxplus', 'rxminus'
                Which rotations you want to include in your train/test data
            nview: int, 18
                How many views you want to sample for each rotation, up to 18
            ratio: List[int], 25, 50, 75, 100
                Which zoom-out ratios you want to include in your train/test data
            mode: str, 'sv', 'mv', 'sp'
                Which mode you want to load your data
                'sv' - single view, 'mv' - multi view, 'sp' - sphere view
            transform:
                The torch transforms after you load your data
        """
        self.root = root
        self.tr = tr
        self.nview = nview
        self.view_index = np.linspace(0, 17, nview, endpoint=True, dtype=np.int) if nview > 0 else None
        self.ratio = ratio
        self.mode = mode
        self.dataset = dataset

        self.all_files = sorted(glob(f'{root}/*/{dataset}/*/*/*.png'))
        self.all_df = self._get_df(read_csv=True)
        self.df = self._filter()  # self.df will be then final data for train/test after filtering

        self.transform = transform

    def _get_df(self, read_csv):
        if read_csv:
            return pd.read_csv('squares_same_nview.csv', index_col=0)
        else:
            df = pd.DataFrame(
                list(map(self._get_img_info, self.all_files)),
                columns=['path', 'ca', 'no', 'tr', 'fr', 'ratio'],
            )
            view_index = [i for i in range(18) for _ in range(4)] * (len(self.all_files) // 18 // 4)
            df['view_index'] = view_index
            df.to_csv('squares_same_nview.csv')
            return df

    def _filter(self):
        df = self.all_df
        df = df[df['tr'].isin(self.tr)] if self.tr is not None else df
        df = df[df['view_index'].isin(self.view_index)] if self.view_index is not None else df
        df = df[df['ratio'].isin(self.ratio)] if self.ratio is not None else df
        return df.reset_index().drop('index', axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        info = self.df.loc[index]
        label = conf.ALL_CA.index(info.ca)

        if self.mode == 'sv':
            img = imageio.imread(info.path)
            if self.transform is not None:
                img = self.transform(img)
            path = info.path
        elif self.mode == 'mv':
            mv_info = self.df[
                (self.df.ca == info.ca) &
                (self.df.no == info.no) &
                (self.df.tr == info.tr) &
                (self.df.ratio == info.ratio)
            ]
            imgs = [imageio.imread(p) for p in mv_info.path]
            if self.transform is not None:
                imgs = [self.transform(img) for img in imgs]

            img = torch.stack(imgs).float()
            path = '>'.join(mv_info.path)
        elif self.mode == 'sp':
            mv_info = self.df[
                (self.df.ca == info.ca) &
                (self.df.no == info.no) &
                (self.df.tr == info.tr) &
                (self.df.ratio == info.ratio)
            ]
            imgs = [imageio.imread(p) for p in mv_info.path]

            # Build the sphere representation steps
            size = imgs[0].shape[0]
            l = size // 2
            step = math.ceil(size / self.nview)
            widths = []
            while size > step:
                widths.append(step)
                size -= step
            widths.append(size)

            sp_imgs = []
            for img, w in zip(imgs, widths):
                sp_imgs.append(img[:, l: l + w])
            img = np.hstack(sp_imgs)
            if self.transform is not None:
                img = self.transform(img)
            path = '>'.join(mv_info.path)
        else:
            raise ValueError(f'invalid mode {self.mode}')

        return img, label, path

    @staticmethod
    def _get_img_info(path):
        s = path.split('/')
        ca = s[-5]
        no = s[-3]
        tr, fr = s[-2].split('_')
        ratio, _ = s[-1].split('.')
        return path, ca, int(no), tr, int(fr), int(ratio)