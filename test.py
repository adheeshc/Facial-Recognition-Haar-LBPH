import cv2
import numpy as np
import os
import face as fd
import copy

test_image=cv2.imread('/Test/adheesh4.jpeg')
face_detected, gray_img=fd.detect_face(test_image)
print(f'faces detected :{face_detected}')

faces,faceID=fd.training_labels('/Train')
face_recog=fd.train_classifier(faces,faceID)
face_recog.save('trained_model.yml')

#############
#ONCE TRAINED, YOU CAN COMMENT LINES 11,12,13 AND UNCOMMENT THE 2 LINES BELOW
#############

# face_recog=cv2.face.LBPHFaceRecognizer_create()
# face_recog.read('/trained_model.yml')


names={0:"Random",1:"Adheesh"}

for face in face_detected:
	(x,y,w,h)=face
	roi=gray_img[y:y+h,x:x+w]
	label, confidence= face_recog.predict(roi)
	print(f'the image is {label} with confidence of {confidence}')
	fd.bbox(test_image,face)
	prediction=names[label]
	if confidence>60:
		continue
	fd.label_bbox(test_image,prediction,x,y)

test_image=cv2.resize(test_image,None,fx=0.4,fy=0.4)
cv2.imshow("frame",test_image)
cv2.waitKey(0)
cv2.destroyAllWindows

