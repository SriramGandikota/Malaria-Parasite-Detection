import os
import cv2
from tqdm import tqdm
import numpy as np
par = os.listdir(os.getcwd()+"/data/train/Parasitized")
uni = os.listdir(os.getcwd()+"/data/train/Uninfected")



def getContours(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.Canny(img,100,200)
    ret, thresh = cv2.threshold(imgray, 125, 255, 0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)#cv2.CHAIN_APPROX_TC89_KCOS)
    p = []
    for contour in contours:
        p.append(cv2.contourArea(contour))
    p.sort()
    if(len(p)==0):
        return (0,0)
    if(len(p)<2):
        return (p[-1],0)
    return (p[-1],p[-2])


X = []
Y = []

for file in tqdm(par):
    img = cv2.imread(os.getcwd()+"/data/train/Parasitized/{}".format(file))
    X.append(getContours(img))
    Y.append(1)

for file in tqdm(uni):
    img = cv2.imread(os.getcwd()+"/data/train/Uninfected/{}".format(file))
    X.append(getContours(img))
    Y.append(0)
X = np.asarray(X)
Y = np.asarray(Y)
np.save('contours.npy',X)
np.save('contours_labels.npy',Y)
