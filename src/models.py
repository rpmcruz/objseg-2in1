import torch
import torchvision
import objdetect as od
import unet

class SegHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(weights='DEFAULT')
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.scores = torch.nn.Conv2d(2048, 1, 1)
        self.bboxes = torch.nn.Conv2d(2048, 4, 1)
        tconvs = []
        prev = 2048
        for next in [512, 256, 128, 64, 32]:
            tconvs.append(torch.nn.ConvTranspose2d(prev, next, 3, 2, 1, 1))
            prev = next
        tconvs.append(torch.nn.Conv2d(prev, 1, 1))
        self.seg = torch.nn.Sequential(*tconvs)

    def forward(self, x, threshold=0.5):
        x = self.backbone(x)
        scores = self.scores(x)
        bboxes = self.bboxes(x)
        seg = self.seg(x)
        if not self.training:
            # when in evaluation mode, convert the output grid into a list of bboxes
            scores = torch.sigmoid(scores)
            hasobjs = scores >= threshold
            scores = od.grid.inv_scores(hasobjs, scores)
            bboxes = od.grid.inv_offset_logsize_bboxes(hasobjs, bboxes)
            #bboxes = od.post.NMS(scores, bboxes)
            return bboxes, seg
        return scores, bboxes, seg

class ObjAfterSeg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = unet.UNet(3)
        self.scores = torch.nn.Conv2d(2048, 1, 1)
        self.bboxes = torch.nn.Conv2d(2048, 4, 1)

    def forward(self, x):
        seg, x = self.unet(x)
        x = torch.nn.functional.interpolate(x, scale_factor=32)
        scores = self.scores(x)
        bboxes = self.bboxes(x)
        if not self.training:
            # when in evaluation mode, convert the output grid into a list of bboxes
            scores = torch.sigmoid(scores)
            hasobjs = scores >= threshold
            scores = od.grid.inv_scores(hasobjs, scores)
            bboxes = od.grid.inv_offset_logsize_bboxes(hasobjs, bboxes)
            #bboxes = od.post.NMS(scores, bboxes)
            return bboxes, seg
        return scores, bboxes, seg
