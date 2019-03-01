import  numpy as np
import math
import normalization
from skimage import io, data, color
import pickle
import cv2
# eachpic = '../data1/1/new/0.jpg'
# im = io.imread(eachpic, as_gray=False)
# im = color.rgb2gray(im)
# rows, cols = im.shape
# for i in range(rows):
#     for j in range(cols):
#         im[i, j] = 0 if im[i, j] <= 0.5 else 1
# im = im.astype(np.float32)
# with open('../data1/1/training_data.pkl', 'rb') as handle:
#     da = pickle.load(handle)
#
# output = im - da[2][100].astype(np.float32)
# output = 1- output
# print output
# cv2.imshow("Image", output)
# cv2.waitKey(0)

# eachpic = '../data1/1/new/120.jpg'
# im = io.imread(eachpic, as_gray=False)
# cv2.imshow('frame', im)
# cv2.waitKey(0)
# im = color.rgb2gray(im)

# cap = cv2.VideoCapture('../data1/3/origin.avi')
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.createBackgroundSubtractorMOG2()
#
#
# while True:
#     ret, frame = cap.read()
#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#     im, contours, hir = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for c in contours:
#         per = cv2.arcLength(c, True)
#         if per > 188:
#             x,y,w,h = cv2.boundingRect(c)
#             cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
#
#     cv2.imshow('frame', frame)
#     cv2.imshow('fgmask', fgmask)
#
#     k = cv2.waitKey(30) & 0xff
#     if k ==27:
#         break
# cap.release()
# cv2.destroyAllWindows()
