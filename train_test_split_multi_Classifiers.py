from sklearn import datasets
iris = datasets.load_iris()

import matplotlib.pyplot as plt

X= iris.data
y = iris.target


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5)


from sklearn import tree

my_classifier = tree.DecisionTreeClassifier()

#we can use different classifier

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

#notice that all methods under classifier are the same

my_classifier.fit(X_train, y_train)
prediction = my_classifier.predict(X_test)
print(prediction)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, prediction))
