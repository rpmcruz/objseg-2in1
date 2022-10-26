# Obj and Seg: 2in1

Multi-task object detection + semantic segmentation. This code uses the [objdetect package](https://github.com/rpmcruz/objdetect) to build the object detection code.

A couple of simplifications:
1. Only detecting bounding-boxes and segmentations of cars.
2. Only training with cases where we have both bounding-boxes and segmentations.

Two architectures are here implemented:

**Architecture SegHead:** We predict segmentation on top of the object grid.

![](doc/seg-head.png)

**Architecture ObjAfterSeg:** We predict objects on top of the penultimate layer from the segmentation model. (The reason why the penultimate layer is not used, instead of the final output layer, is that the penultimate layer has the same resolution as the final layer but is richer in features, so hopefully it improves results.)

![](doc/obj-after-seg.png)

-- Ricardo Cruz &lt;rpcruz@fe.up.pt&gt;
