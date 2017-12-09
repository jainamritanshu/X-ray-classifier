from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

neighbors = 5

def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	else:
		cv2.normalize(hist, hist)

	return hist.flatten()

def extract_sift(image):

	gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(gray,None)
	print des
	return des

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

print("describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
rawImages = []
features = []
labels = []
sift_features = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-2:-1][0]

	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	sift_feature = extract_sift(image)
	print hist

	rawImages.append(pixels)
	features.append(hist)
	sift_features.append(sift_feature)
	labels.append(label)

	print("processed {}/{}".format(i, len(imagePaths)))

rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

print("evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=neighbors)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("raw pixel accuracy: {:.2f}%".format(acc * 100))

predicted = model.predict(testRI)
report = classification_report(testRL, predicted)
print(report)

print("evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=neighbors)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("histogram accuracy: {:.2f}%".format(acc * 100))

predicted = model.predict(testFeat)
report = classification_report(testLabels, predicted)
print(report)