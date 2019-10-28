# Facial-Recognition-Haar-LBPH

## **PROJECT DESCRIPTION**

Basic facial recognition implemented using Haar Cascades and Local Binary Pattern Histograms

## **DEPENDANCIES**

- Python 3
- OpenCV
- Numpy
- os (built-in)
- Copy (built-in)

## **FILE DESCRIPTION**

[Face.py]() - Main facial recognition code

[Test.py]() - Tester file to train and test on images

[Init.py]() - real time face recognition

## **RUN INSTRUCTIONS**

- Make sure all dependancies are met
- Comment/Uncomment as reqd

### **CREATING DATASETS**
-Create 2 directories - Train and Test
- In Test folder, have around 10 images of the images you want to recognize (eg. images of you) called positive set and 10 images to check for misclassification (eg. images of NOT you) called negative set
- In Train folder, create 2 directories - 0 and 1
  - In directory 0, keep all negative set images (NOT you) and in directory 1 keep all positive set images (YOU)
  
 Make sure you have around a 100 or so images for good results

