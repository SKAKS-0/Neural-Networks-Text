import numpy as np
import pandas as pd
from matplotlib import style
import sklearn
import tensorflow as tf
import keras
from sklearn import linear_model
import pickle
import matplotlib.pyplot as pyplot
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder as le

data = pd.read_csv("breast-cancer-wisconsin.csv", sep = ";")
predict = "Diagnosis"
label = le.fit_transform(list[data[predict]])
data.drop(predict, axis=1, inplace=True)
x = np.array(data.drop[predict],1)
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

best = 0

for _ in range(30):

    x_train,x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    if acc > best:
        best = acc
        with open("BC.pickle", "wb") as BC:
            pickle.dump(linear, BC)
        print(acc)


    use_it = linear.predict(x_test)
    for i in range(len(use_it)):
        print(x_test[i],y_test[i],use_it[i])


model = open("BC.pickle", "rb")
md = pickle.load(model)


