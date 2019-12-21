import os
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def calc_accuracy(pred,test):
    cnt = 0
    ones = 0.0
    zeroes= 0.0
    for i in range(len(pred)):
        if(pred[i]!=test[i]):
            cnt+=1
            if(test[i]==1):
                ones+=1
            if(test[i]==0):
                zeroes+=1
    return ((len(pred)-cnt+1.0-1.0)*100/(len(pred)),zeroes/len(pred),ones/len(pred))

X = np.load('contours.npy')
Y = np.load('contours_labels.npy')

train_X,test_X,train_Y,test_Y = train_test_split(X,Y,random_state=42,test_size=0.2)

model = LogisticRegression()
model.fit(train_X,train_Y)
pred = model.predict(test_X)
print(calc_accuracy(pred,test_Y))

model1 = SVC()
model1.fit(train_X,train_Y)
pred = model1.predict(test_X)
print(calc_accuracy(pred,test_Y))


model2 = KNeighborsClassifier()
model2.fit(train_X,train_Y)
pred= model2.predict(test_X)
print(calc_accuracy(pred,test_Y))

