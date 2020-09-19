import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split


##Step 1: Get Data from CSV
df = pd.read_csv("digit_recognizer_dataset.csv")
print(df.shape)


##Step 2: Seperate Labels and Features
X = df.drop(['label'],axis=1)
Y = df['label']


##Step 3: Make sure you have the correct Feature / label combination in training

X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 38)

##Step 4: Build a Model and Save it

model = svm.SVC()
model.fit(X_train,Y_train)

joblib.dump(model,"svm_4label_data")
# print("Accuracy score",metrics.accuracy_score())
print("Accuracy score",model.score(X_test,Y_test))
 

##Step5 : Print Accuracy 
