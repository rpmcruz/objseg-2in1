import argparse
parser = argparse.ArgumentParser()
parser.add_argument('architecture', choices=['SegHead', 'ObjAfterSeg'])
parser.add_argument('output')
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import objdetect as od
from time import time
import data, models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################## DATA ##########################

transform = A.Compose([
    A.Resize(int(256*1.1), int(256*1.1)),
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=1),
    A.Normalize(),
    ToTensorV2(),
], bbox_params=A.BboxParams('pascal_voc', []))

ds = data.KITTI('/data', 'train', transform)
ds = torch.utils.data.Subset(ds, range(100))  # DEBUG
tr = torch.utils.data.DataLoader(ds, 32, True, num_workers=0, pin_memory=True,
    collate_fn=data.collate_fn)

########################## MODEL ##########################

model = getattr(models, args.architecture)
model = model().to(device)
scores_loss = torchvision.ops.focal_loss.sigmoid_focal_loss
bboxes_loss = torch.nn.MSELoss(reduction='none')
seg_loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

########################## TRAIN ##########################

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    for imgs, masks, bboxes in tr:
        imgs = imgs.to(device)
        preds_scores, preds_bboxes, preds_masks = model(imgs)

        slices = od.grid.slices_center_locations(8, 8, bboxes)
        scores = od.grid.scores(8, 8, slices, device=device)
        bboxes = od.grid.offset_logsize_bboxes(8, 8, slices, bboxes, device=device)
        masks = masks.to(device)[:, None].float()

        loss_value = \
            scores_loss(preds_scores, scores).mean() + \
            (scores * bboxes_loss(preds_bboxes, bboxes)).mean() + \
            seg_loss(preds_masks, masks)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        avg_loss += float(loss_value) / len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {avg_loss}')
