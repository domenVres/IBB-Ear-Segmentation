import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN
from engine import train_one_epoch
import utils
import torch
import numpy as np
from PIL import Image


class CNNEarDetector:
    def __init__(self, hidden_layer=256, backbone=None):
        num_classes = 2

        # Resnet backbone
        if backbone is None:
            # load a model pre-trained on COCO
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

            # get number of input features for the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # now get the number of input features for the mask classifier
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = hidden_layer
            # and replace the mask predictor with a new one
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                               hidden_layer,
                                                               num_classes)

        elif backbone == "mobile_net":
            # load a pre-trained model for classification and return
            # only the features
            bbone = torchvision.models.mobilenet_v2(pretrained=True).features
            # FasterRCNN needs to know the number of
            # output channels in a backbone. For mobilenet_v2, it's 1280
            # so we need to add it here
            bbone.out_channels = 1280

            # let's make the RPN generate 5 x 3 anchors per spatial
            # location, with 5 different sizes and 3 different aspect
            # ratios. We have a Tuple[Tuple[int]] because each feature
            # map could potentially have different sizes and
            # aspect ratios
            anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                               aspect_ratios=((0.5, 1.0, 2.0),))

            # let's define what are the feature maps that we will
            # use to perform the region of interest cropping, as well as
            # the size of the crop after rescaling.
            # if your backbone returns a Tensor, featmap_names is expected to
            # be [0]. More generally, the backbone should return an
            # OrderedDict[Tensor], and in featmap_names you can choose which
            # feature maps to use.
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                            output_size=7,
                                                            sampling_ratio=2)

            mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                                 output_size=14,
                                                                 sampling_ratio=2)

            self.model = MaskRCNN(bbone,
                                  num_classes=2,
                                  rpn_anchor_generator=anchor_generator,
                                  box_roi_pool=roi_pooler,
                                  mask_roi_pool=mask_roi_pooler)

        self.optimizer = None

    def set_otpimizer(self, learning_rate=0.001, momentum=0.9, weight_decay=0.0005):
        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    def train(self, data_train, batch_size=2, step_size=3, gamma=0.1, num_epochs=10):
        # train on the GPU or on the CPU, if a GPU is not available
        device = None
        if torch.cuda.is_available():
            print("Training model on GPUs")
            device = torch.device('cuda')
        else:
            print("Training model on CPUs")
            device = torch.device('cpu')

        data_loader = torch.utils.data.DataLoader(
            data_train, batch_size=batch_size, shuffle=True, num_workers=2,
            collate_fn=utils.collate_fn)

        self.model.to(device)

        if self.optimizer is None:
            self.set_otpimizer()

        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                       step_size=step_size,
                                                       gamma=gamma)

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, self.optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()

    def save(self, root, filename):
        torch.save({"model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   root+"/detectors/your_super_detector/"+filename+".pt")

    def load(self, root, filename):
        if torch.cuda.is_available():
            checkpoint = torch.load(root+"/detectors/your_super_detector/"+filename+".pt")
        else:
            checkpoint = torch.load(root + "/detectors/your_super_detector/" + filename + ".pt", map_location=torch.device('cpu'))

        self.model.load_state_dict(checkpoint['model_state_dict'])

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        """if torch.cuda.is_available():
            print("Model moved to GPU")
            self.model.to(torch.device('cuda'))"""

    def store_predictions(self, test_data_loader, root, folder):
        data_iter = iter(test_data_loader)

        self.model.eval()
        i = 0
        for images, targets in data_iter:
            if i % 10 == 0:
                print(f"Iteration {i}")
            images = list(image for image in images)
            outputs = self.model(images)
            outputs = outputs[0]["masks"]
            masks = [mask.to(torch.device('cpu')).detach().numpy()[0] for mask in outputs]
            mask = np.zeros(masks[0].shape)
            for m in masks:
                mask = np.logical_or(mask, m > 0.5)
            mask = mask.astype("uint8")
            mask *= 255
            im = Image.fromarray(mask, mode='L').convert('1')
            im.save(root+"/detectors/your_super_detector/"+folder+f"/{i+1:04d}"+".png")

            i += 1

    def __call__(self, *args):
        return self.model(*args)

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()