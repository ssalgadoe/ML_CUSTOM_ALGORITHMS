from scipy.spatial import distance
from sklearn import datasets
iris = datasets.load_iris()

import matplotlib.pyplot as plt
import random

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        prediction = []
        for row in X_test:
            label = self.closest(row)
            prediction.append(label)
        return prediction

    def closest(self, row):
        best_dist = euc(row,self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]        
        
X= iris.data
y = iris.target
 

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5)


from sklearn import tree

#decision tree
#my_classifier = tree.DecisionTreeClassifier()

#we can use different classifier (example K-Neighbours)

from sklearn.neighbors import KNeighborsClassifier
#my_classifier = KNeighborsClassifier()

#notice that all methods under classifier are the same

# my own classifier (K-Neighbours)
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)
prediction = my_classifier.predict(X_test)
#print(prediction)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, prediction))
