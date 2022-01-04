# IBB-Ear-Segmentation
 
This is a code for ear segmentation/detection assignment, which was a part of Image based biometry course at Faculty of Computer and Information Science, University of Ljubljana.

The assignment was implemented on AWE dataset [[1]](#1). I compared Viola-Jones detector with Mask R-CNN using ResNet50 and MobileNetV2 as backbones. The starting point for my CNN was [[2]](#2). I also used different image preprocessing techniques.

I obtained the following results:
![](results.png)

## References

<a id="1">[1]</a>
http://ears.fri.uni-lj.si/datasets.html#awe-full

<a id="2">[2]</a>
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html