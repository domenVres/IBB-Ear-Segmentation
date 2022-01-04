import cv2, sys, os
import numpy as np

class CascadeDetector:
	# This example of a detector detects faces. However, you have annotations for ears!

	def __init__(self):
		self.cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'lbpcascade_frontalface.xml'))
		# self.cascade = cv2.CascadeClassifier("cascades/haarcascade_mcs_leftear.xml")
		# self.cascade = cv2.CascadeClassifier("cascades/haarcascade_mcs_rightear.xml")

	def detect(self, img):
		det_list = self.cascade.detectMultiScale(img, 1.05, 1)
		return det_list


class CascadeEarDetector:
	def __init__(self):
		self.lear_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_leftear.xml'))
		self.rear_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_rightear.xml'))

	def detect(self, img):
		lear_list = self.lear_cascade.detectMultiScale(img, 1.05, 1)
		if lear_list == ():
			lear_list = np.array([[]]).reshape(0, 4).astype("int32")
		rear_list = self.rear_cascade.detectMultiScale(img, 1.05, 1)
		if rear_list == ():
			rear_list = np.array([[]]).reshape(0, 4).astype("int32")
		return np.concatenate([lear_list, rear_list], axis=0)

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(f"../../data/ears/test/{fname}")
	"""detector = CascadeDetector()
	detected_loc = detector.detect(img)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(fname.rstrip(".png") + '_detected.jpg', img)"""

	detector = CascadeEarDetector()
	detected_loc = detector.detect(img)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(fname + '_detected_ear.jpg', img)