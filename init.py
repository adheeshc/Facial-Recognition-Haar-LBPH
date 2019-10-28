import cv2
import numpy as np
import os
import face as fd
import copy

face_recog=cv2.face.LBPHFaceRecognizer_create()
face_recog.read('/trained_model.yml')

names={0:"Random",1:"Adheesh"}

cap=cv2.VideoCapture(0)

while (True):
	ret,frame=cap.read()
	test_image=copy.deepcopy(frame)
	face_detected, gray_img=fd.detect_face(test_image)
	print(f'faces detected :{face_detected}')

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

	#test_image=cv2.resize(test_image,None,fx=0.4,fy=0.4)
	cv2.imshow("frame",test_image)

	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
