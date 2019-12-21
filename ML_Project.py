
# coding: utf-8

# # Malaria Parasite Detection

# In[6]:


'''
The expected directory structure for the given images for running a deep network is 
    --Current folder 
            -- data 
                  |
                  | --Parasite
                       |
                       | --- train
                       |     |
                       |     | -----Uninfected
                       |     | ------Parasitized
                       |------test
 '''


# In[1]:


#importing required libraries
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import datetime as now
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle


# In[2]:


#required helper functions

#Function to get the distance from the dark pink value which also considers the brightness
def dist(RGB):
    return ((RGB[0]-255)**2 + (RGB[1]-20)**2 + (RGB[2]-147)**2)**0.5
#Calculating the maximum varance and distance for a given (i,j) pixel
def neighbor_var(img,i,j):
    ans = -1
    x = [1,0,-1]
    y = [1,0,-1]
    for a in x:
        for b in y:
            if(i+a<img.shape[0] and j+b<img.shape[1] and (a!=0 or b!=0)):
                if(img[i+a][j+b][0]!=0 or img[i+a][j+b][1]!=0 or img[i+a][j+b][2]!=0):
                    ans = max(ans,
                    ((img[i][j][0]-img[i+a][j+b][0])**2 + (img[i][j][1]-img[i+a][j+b][1])**2 + (img[i][j][2]-img[i+a][j+b][2])**2)**0.5)
    return ans
#Accuracy Score
def calc_acc(predict,original):
    count = 0
    for i in range(len(predict)):
        if(predict[i]==original[i]):
            count+=1
    return (count+0.0)/(0.0+len(predict))
def true_positive(predict,original):
    count = 0
    count1 = 0
    for i in range(len(predict)):
        if(predict[i]!=original[i]):
            count+=1
        if(predict[i]==0 and original[i]==1):
            count1+=1
    return (count1+0.0)/(count+0.0)
#Gets the area of two largest contours 
def getContours(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

def extract(img):
    d = 3*(255**2)
    var = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j][0]!=0 or img[i][j][1]!=0 or img[i][j][2]!=0):
                d = min(d,dist(img[i][j]))
                var = max(var,neighbor_var(img,i,j))
    return np.asarray([d,var])


# In[3]:


#global variables 

IMG_SIZE = (130,130,3) #If image requires resizing this is the size
epochs=20 #Number of epochs for deep learning 
#Inorder to use the preprocessed data directly this variable must be kept as False
preprocess = False


# ## Extracting featuers related to the 'pinkess' of a pixel and its variance and using Contour area

# In[4]:


if(preprocess):
    infected = os.listdir(os.getcwd()+"/Parasitized")
    uninfected = os.listdir(os.getcwd()+"/Uninfected")
    X = []
    Y = []

    for file in tqdm(infected):
        img  = cv2.imread(os.getcwd()+"/Parasitized/{}".format(file))
        X.append(img)
        Y.append(0)
    for file in tqdm(uninfected):
        img  = cv2.imread(os.getcwd()+"/Uninfected/{}".format(file))
        X.append(img)
        Y.append(1)
    print("Data Preprocessing.....")
    for img in tqdm(X):
        X.append(extract(img))
    for img in tqdm(X):
        X.append(extract(img))
else:
    X = np.load('variance.npy')
    Y = np.load('variance_Y.npy')
    


# In[5]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


# Model Training 

# In[ ]:


#KNN
if(preprocess):
    KNN = KNeighborsClassifier(n_neighbors = 193)
    KNN.fit(X_train,Y_train)
else:
    KNN = pickle.load(open("KNN.sav", 'rb'))
print(calc_acc(KNN.predict(X_test),Y_test),true_positive(KNN.predict(X_test),Y_test))
    


# In[ ]:


#Support Vector Machine
if(preprocess):
    SVM_cla = SVC(gamma='scale')
    SVM_cla.fit(X_train,Y_train)
else:
    SVM_cla = pickle.load(open("Support_Vector_Machine.sav", 'rb'))
print(calc_acc(SVM_cla.predict(X_test),Y_test))


# In[ ]:


#Logistic Regression
if(preprocess):
    model = LogisticRegression()
    model.fit(X_train,Y_train)
else:
    model = pickle.load(open("LogisticRegression.sav", 'rb'))
print(calc_acc(model.predict(X_test),Y_test))


# ## Extracting the size of area of contours

# In[ ]:


if(preprocess):
    X_c = []
    Y_c = []

    for file in tqdm(par):
        img = cv2.imread(os.getcwd()+"/data/train/Parasitized/{}".format(file))
        X_c.append(getContours(img))
        Y_c.append(1)

    for file in tqdm(uni):
        img = cv2.imread(os.getcwd()+"/data/train/Uninfected/{}".format(file))
        X_c.append(getContours(img))
        Y_c.append(0)
    X_c = np.asarray(X_c)
    Y_c = np.asarray(Y_c)
    np.save('contours.npy',X_c)
    np.save('contours_labels.npy',Y_c)
else:
    X_c = np.load('contours.npy')
    Y_c = np.load('contours_labels.npy')


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X_c,Y_c,test_size=0.2,random_state=42)


# Model Building

# In[ ]:


#KNN
if(preprocess):
    KNN = KNeighborsClassifier(n_neighbors = 193)
    KNN.fit(X_train,Y_train)
else:

    
print(calc_acc(KNN.predict(X_test),Y_test))
    
    


# In[ ]:


#Support Vector Machine
if(preprocess):
    SVM_cla = SVC(gamma='scale')
    SVM_cla.fit(X_train,Y_train)
else:

print(calc_acc(SVM_cla.predict(X_test),Y_test))
    


# In[ ]:


#Logistic Regression
if(preprocess):
    model = LogisticRegression()
    model.fit(X_train,Y_train)
else:

    
print(calc_acc(model.predict(X_test),Y_test))
    


# ## Adopting Deep Learning Apporaches

# Data Pre-Processing

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, LeakyReLU, Input, Flatten,add, MaxPooling2D, BatchNormalization, Input, Dropout, add
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from IPython.display import FileLink
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau


# In[ ]:


DATA_PATH = "data/Parasite/"


# In[ ]:


get_ipython().system('rm -rf train val')


# In[ ]:


os.mkdir("train")
os.mkdir("train/0")
os.mkdir("train/1")


# In[ ]:


par = os.listdir(DATA_PATH+"/train/Parasitized/")
uni = os.listdir(DATA_PATH+"/train/Uninfected/")
uni.remove("Thumbs.db")
par.remove("Thumbs.db")


# In[ ]:



#making directories and storing the pre processed images
for file in tqdm(par):
    img = cv2.imread(DATA_PATH+"/train/Parasitized/{}".format(file))
    img = cv2.resize(img,dsize=(130,130),interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.getcwd()+"/train/1/{}".format(file),img)
for file in tqdm(uni):
    img = cv2.imread(DATA_PATH+"/train/Uninfected/{}".format(file))
    img = cv2.resize(img,dsize=(130,130),interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.getcwd()+"/train/0/{}".format(file),img)


# Deep Neural Network

# In[ ]:


model = Sequential()
model.add(Conv2D(filters=64,kernel_size=(5,5),activation='relu',input_shape=IMG_SIZE))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=16,kernel_size=(5,5),activation='relu'))
model.add(Conv2D(filters=8,kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv2D(filters=4,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid'))
 


# In[ ]:


traingen = ImageDataGenerator(rescale=1./255,rotation_range=15,validation_split=0.2)
train = traingen.flow_from_directory('train/',shuffle=True,batch_size=64,color_mode='rgb',target_size=(130,130),class_mode='binary',subset='training')
val = traingen.flow_from_directory('train/',shuffle=True,batch_size=32,color_mode='rgb',target_size=(130,130),class_mode='binary',subset='validation')


# In[ ]:


lr = ReduceLROnPlateau(monitor='accuracy', factor=0.1,patience=2, min_lr=0.0001)
es = EarlyStopping(monitor='loss',min_delta=0.01,patience=4)
history =model.fit_generator(train,epochs=epochs,steps_per_epoch=551,validation_data = val,callbacks=[lr])


# Residual Network

# In[ ]:


inputs = Input(shape=IMG_SIZE)

convb1_1 = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(inputs)
convb1_2 = Conv2D(filters=32,kernel_size=(5,5),activation='relu')(convb1_1)
maxb1_1 = MaxPooling2D(pool_size=(2,2))(convb1_2)
bn_1 = BatchNormalization()(maxb1_1)

convb2_1 = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(bn_1)
convb2_2 = Conv2D(filters=32,kernel_size=(5,5),activation='relu')(convb2_1)
maxb2_1 = MaxPooling2D(pool_size=(2,2))(convb2_2)
bn_2 = BatchNormalization()(maxb2_1)

convb3_1 = Conv2D(filters=64,kernel_size=(5,5),activation='relu')(bn_2)
convb3_2 = Conv2D(filters=32,kernel_size=(3,3),activation='relu')(convb3_1)
maxb3_1 = MaxPooling2D(pool_size=(2,2))(convb3_2)

convb4_1 = Conv2D(filters=32,kernel_size=(3,3),activation='relu')(maxb3_1)
maxb4_1 = MaxPooling2D(pool_size=(2,2))(convb3_1)
flatten = Flatten()(maxb4_1)

fc_1 = Dense(512,activation='relu')(flatten)
dr_1 = Dropout(0.5)(fc_1)
fc_2 = Dense(256,activation='relu')(dr_1)
bn_3 = BatchNormalization()(fc_2)
fc_3 = Dense(512,activation='relu')(bn_3)
res_2 = add([fc_1,fc_3])
dr_2 = Dropout(0.25)(res_2)
fc_4 = Dense(256,activation='relu')(dr_2)
res_2 = add([fc_2,fc_4])
fc_5 = Dense(64,activation='relu')(res_2)
predictions = Dense(1,activation='sigmoid')(fc_5)

resmodel = Model(inputs=inputs,outputs=predictions)


# In[ ]:


lr = ReduceLROnPlateau(monitor='accuracy', factor=0.1,patience=2, min_lr=0.0001)
es = EarlyStopping(monitor='loss',min_delta=0.01,patience=4)
history =resmodel.fit_generator(train,epochs=epochs,steps_per_epoch=551,validation_data = val,callbacks=[lr])


# In[ ]:


plt.plot(history.history['loss'])
plt.show()


# In[ ]:


#Exractign test data
testfiles = os.listdir(DATA_PATH+"test/")
test = []
for file in tqdm(testfiles):
    img = cv2.imread(DATA_PATH+"test/{}".format(file))
    img = cv2.resize(img,dsize=(130,130))
    test.append(img)
test = np.asarray(test)
test = test.astype('float32')
test /= 255.0


# In[ ]:


#predicting values of the model 
#change model to resmodel for predicting residual networks output
pred = model.predict(test)


# In[ ]:


#making the csv submission file
from time import time
fl = "submission-{}.csv".format(time())
f = open(fl,"w")
f.write("Name,Label\n")
for i in range(0,len(testfiles)):
    ans= 0
    if(pred[i]>0.5):
        ans = 1
    f.write("{},{}\n".format(testfiles[i],ans))
f.close()


# In[ ]:


# the models are there in the folder given 


# In[ ]:


#the weights are there in submission_res_model_weights.h5 and model is in submission_res_model.h5
filename = "submission_res_model.h5" # for loading the model directly
model = keras.models.load_model("{}.h5".format(filename))

