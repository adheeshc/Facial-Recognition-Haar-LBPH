import numpy as np
import cv2
import os

def detect_face(test_img):
	gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
	face_haar=cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')
	faces=face_haar.detectMultiScale(gray_img,scaleFactor=1.3141526,minNeighbors=5)

	return faces, gray_img

def training_labels(directory):
	faces=[]
	faceID=[]
	for root, subdir, filename in os.walk(directory):
		for f in filename:
			#SKIP SYSTEM/HIDDEN FILES
			if f.startswith("."):
				print(f"Skipped system file: {f}")
				continue

			#GET IMAGE PATH
			id = os.path.basename(root)
			img_path=os.path.join(root,f)
			print(f'img_path : {img_path}')
			print(f'id : {id}')
			test_img=cv2.imread(img_path)

			#CHECK FOR PATH ERROR
			if test_img is None:
				print(f"Image not loaded from {img_path}")
				continue
			
			#CHECK FOR SINGLE FACE IMAGES
			faces_rect,gray_img=detect_face(test_img)
			if len(faces_rect)!=1:
				continue

			#DEFINE ROI
			(x,y,w,h)=faces_rect[0]
			roi=gray_img[y:y+w,x:x+h]
			faces.append(roi)
			faceID.append(int(id))


	return faces,faceID


def train_classifier(faces,faceID):

	#USING LOCAL BINARY PATTERN HISTOGRAM
	face_recog=cv2.face.LBPHFaceRecognizer_create()
	face_recog.train(faces,np.array(faceID))
	return face_recog


def bbox(test_img,faces):
	(x,y,w,h) = faces
	cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

def label_bbox(test_img,text,x,y):
	cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0),1)


