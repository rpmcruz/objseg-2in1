import torch
import numpy as np
from skimage.io import imread
import pandas as pd
import os

def collate_fn(batch):
    imgs = torch.stack([torch.as_tensor(d['image']) for d in batch])
    masks = torch.stack([torch.as_tensor(d['mask']) for d in batch])
    bboxes = [torch.as_tensor(d['bboxes']) for d in batch]
    return imgs, masks, bboxes

def train_test_split(x, split):
    rand = np.random.RandomState(123)
    ix = rand.choice(len(x), len(x), False)
    ix = ix[:int(len(x)*0.75)] if split == 'train' else ix[int(len(x)*0.75):]
    return [x[i] for i in ix]

class KITTI(torch.utils.data.Dataset):
    # loads both bounding boxes and semantic segmentation for cars only
    def __init__(self, root, split, transform=None):
        assert split in ('train', 'test')
        self.img_root = os.path.join(root, 'kitti', 'object', 'training', 'image_2')
        self.obj_root = os.path.join(root, 'kitti', 'object', 'training', 'label_2')
        self.seg_root = os.path.join(root, 'kitti', 'semantics', 'training', 'semantic')
        self.files = sorted(os.listdir(self.seg_root))
        self.files = [f[:-7] for f in self.files]
        self.files = train_test_split(self.files, split)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        image = imread(os.path.join(self.img_root, fname + '.png'))
        df = pd.read_csv(os.path.join(self.obj_root, fname + '.txt'),
            sep=' ', header=None, usecols=[0, 4, 5, 6, 7])
        # bboxes format = (x1y1x2y2) absolute (same as pascal_voc)
        bboxes = df.loc[df.iloc[:, 0] == 'Car'].iloc[:, [1, 2, 3, 4]].to_numpy(np.float32)
        mask = imread(os.path.join(self.seg_root, fname + '_10.png'), True)
        mask = (mask == 26).astype(np.uint8)
        d = {'image': image, 'bboxes': bboxes, 'mask': mask}
        if self.transform:
            d = self.transform(**d)
        return d
