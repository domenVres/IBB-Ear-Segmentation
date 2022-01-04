import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
from tqdm.auto import tqdm

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']


    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
            return annot


    def run_evaluation(self, preprocessing=None):
        """
        :param preprocessing: either None (no preprocessing), "histogram" (histogram equalization), "edge" (edge
        enhancement) or "sharpen" (image sharpening)
        :return:
        """

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        f1_arr = []
        preprocess = Preprocess()
        eval = Evaluation()
        
        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        # import detectors.your_super_detector.detector as super_detector
        cascade_detector = cascade_detector.CascadeEarDetector()
        

        for im_name in tqdm(im_list):
            
            # Read an image
            img = cv2.imread(im_name)

            # Apply some preprocessing
            if preprocessing == "histogram":
                img = preprocess.histogram_equalization_rgb(img) # This one makes VJ worse
            elif preprocessing == "edge":
                img = preprocess.edge_enhancement(img)
            elif preprocessing == "sharpen":
                img = preprocess.image_sharpening(img)

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            prediction_list = cascade_detector.detect(img)

            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)

            # Only for detection:
            p, gt = eval.prepare_for_detection(prediction_list, annot_list)
            
            iou = eval.iou_compute(p, gt)
            iou_arr.append(iou)
            f1_arr.append(eval.f1_compute(p, gt))

        miou = np.average(iou_arr)
        mf1 = np.average(f1_arr)
        print("\n")
        print("Average IOU:", f"{miou:.2%}")
        print("Average F1-score:", f"{mf1:.2}")
        print("\n")

    def evaluate_cnn(self, folder):
        im_list = sorted(os.listdir(os.path.join(os.getcwd(), "data/ears/test")))
        iou_arr = []
        f1_arr = []
        preprocess = Preprocess()
        eval = Evaluation()

        for im_name in tqdm(im_list):
            # Read an annotation mask and change it to boolean
            img_mask = cv2.imread("data/ears/annotations/segmentation/test/"+im_name)
            img_mask = img_mask > 0

            # Read prediction and change it to boolean
            pred = cv2.imread(f"detectors/your_super_detector/{folder}/" + im_name)
            pred = pred > 0

            iou_arr.append(eval.iou_compute(pred, img_mask))
            f1_arr.append(eval.f1_compute(pred, img_mask))

        miou = np.average(iou_arr)
        mf1 = np.average(f1_arr)
        print("\n")
        print("Average IOU:", f"{miou:.2%}")
        print("Average F1-score:", f"{mf1:.2}")
        print("\n")


if __name__ == '__main__':
    ev = EvaluateAll()
    """print("---------- Viola-Jones ----------")
    print("Without preprocessing:")
    ev.run_evaluation()
    print("Histogram equalization:")
    ev.run_evaluation(preprocessing="histogram")
    print("Edge enhancement:")
    ev.run_evaluation(preprocessing="edge")
    print("Image sharpening:")
    ev.run_evaluation(preprocessing="sharpen")
    print("---------------------------------\n")

    print("---------- Mask R CNN ----------")
    ev.evaluate_cnn("predictions_new")
    print("Histogram equalization:")
    ev.evaluate_cnn("predictions_histogram")
    print("Edge enhancement:")
    ev.evaluate_cnn("predictions_edge")
    print("Image sharpening:")
    ev.evaluate_cnn("predictions_sharpening")"""
    print("Model trained on more epochs:")
    ev.evaluate_cnn("predictions_best")
    print("---------------------------------\n")

    """print("---------- MobileNet Mask R CNN ----------")
    ev.evaluate_cnn("predictions_mobile")
    print("Histogram equalization:")
    ev.evaluate_cnn("predictions_mobile_equalization")
    print("Edge enhancement:")
    ev.evaluate_cnn("predictions_mobile_edge")
    print("Image sharpening:")
    ev.evaluate_cnn("predictions_mobile_sharpening")
    print("Model trained on more epochs:")
    ev.evaluate_cnn("predictions_mobile_best")
    print("---------------------------------\n")"""