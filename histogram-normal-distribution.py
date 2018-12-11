import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





mu, sigma = 5, 1
mu, sigma = 5.0, 0.1 # mean and standard deviation
X1 = np.random.normal(5.0, 2.0, 10)
Y1 = np.random.normal(5.0, 2.0, 10)

X2 = np.random.normal(-5.0, 2.0, 10)
Y2 = np.random.normal(-8.0, 2.0, 10)


#image = np.random.rand(30, 30)
#plt.imshow(image, cmap=plt.cm.gray)
#plt.colorbar()
#plt.show()

plt.scatter(X1,Y1, s=100, color='r')
plt.scatter(X2,Y2, s=100, color='b')
plt.show()






