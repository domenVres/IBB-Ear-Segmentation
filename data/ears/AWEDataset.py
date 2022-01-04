import os
import numpy as np
import torch
from PIL import Image


class AWEDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # These are defined in child classes
        self.imgs = []
        self.masks = []
        self.boxes = []

    def __getitem__(self, path):
        # load images, masks and bounding boxes
        idx = path["idx"]
        img_path = path["image"]
        mask_path = path["mask"]
        box_path = path["box"]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        boxes = []
        with open(box_path) as f:
                lines = f.readlines()
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    x0, y0, w, h = (int(i) for i in l_arr)
                    boxes.append([x0, y0, x0+w, y0+h])

        # Econde as 3D boolean array
        mask = np.array(mask)[None, :, :] == 255

        # convert everything into a torch.Tensor
        num_ears = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_ears,), dtype=torch.int64)
        masks = torch.as_tensor(mask, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_ears,), dtype=torch.int64)

        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class AWETrainSet(AWEDataset):
    def __init__(self, root, transforms):
        super().__init__(root, transforms)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "annotations/segmentation/train"))))
        self.boxes = list(sorted(os.listdir(os.path.join(root, "annotations/detection/train_YOLO_format"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "train", self.imgs[idx])
        mask_path = os.path.join(self.root, "annotations/segmentation/train", self.masks[idx])
        box_path = os.path.join(self.root, "annotations/detection/train_YOLO_format", self.boxes[idx])

        path = {"idx": idx, "image": img_path, "mask": mask_path, "box": box_path}

        return super().__getitem__(path)

    def __len__(self):
        return len(self.imgs)


class AWETestSet(AWEDataset):
    def __init__(self, root, transforms):
        super().__init__(root, transforms)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "test"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "annotations/segmentation/test"))))
        self.boxes = list(sorted(os.listdir(os.path.join(root, "annotations/detection/test_YOLO_format"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "test", self.imgs[idx])
        mask_path = os.path.join(self.root, "annotations/segmentation/test", self.masks[idx])
        box_path = os.path.join(self.root, "annotations/detection/test_YOLO_format", self.boxes[idx])

        path = {"idx": idx, "image": img_path, "mask": mask_path, "box": box_path}

        return super().__getitem__(path)

    def __len__(self):
        return len(self.imgs)
