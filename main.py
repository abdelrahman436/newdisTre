import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
data = load_iris()

X = data.data
y = data.target

print("X list")
print(X)
print('\n')
print('y list')
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=10)

"""
print("x_train")    
print(X_train)
print('\n')
print('x_test')
print(X_test)
print('\n')      
print("y_train")
print(y_train)
print('\n')
print("y_test")
print(y_test)
"""

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

predictionX_train = clf.predict(X_train)
#print(predictionX_train)
predictionX_test = clf.predict(X_test)
#print('\n')
#print(predictionX_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictionX_test))

print(np.mean(X_test))
print(np.mean(y_test))
print(np.mean(y_train))
print(np.mean(X_train))
print(np.std(X_test))
print(np.std(y_test))
print(np.std(y_train))
print(np.std(X_train))

feature_importance = pd.DataFrame(clf.feature_importances_,index=[1,2,3,4])
print(feature_importance)

feature_importance.head(10).plot(kind="bar")
plt.show()

from sklearn import tree

tree.plot_tree(clf, filled=True)
plt.show()
