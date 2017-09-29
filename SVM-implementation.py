import numpy as np
from sklearn import preprocessing,cross_validation,neighbors , svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True) #Make the empty data points as outliers !
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'],1)) # Features are basically everything but class feature , because it is our the features that we want to predict !
y = np.array(df['class']) #Label is just the class because we want to predict it only .

X_train , X_test, y_train ,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)
example_measures = np.array([[1,1,2,1,2,2,1,2,1],[4,2,1,1,1,2,3,2,1]]) # Try on an axample , try to find the 2(benign) or 4 (malignant)
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)