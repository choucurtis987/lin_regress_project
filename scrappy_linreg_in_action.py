# dataset used:
# https://archive.ics.uci.edu/ml/datasets/Auto+MPG

# In this script, I use my custom linear regression model to predict 'mpg' based
# on a car's weight

import sklearn
import matplotlib.pyplot as plt
from scrappy_linreg import linreg

# reading the data and cleaning the data:
df = pd.read_csv('auto-mpg.data',header=None,delimiter=r"\s+")
df.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin', 'car name']
df = df.drop('car name', axis=1)
df['horsepower'] = df['horsepower'].replace('?',-99999)
df = df.astype('float64')

# converting pandas series to numpy arrays and shuffling data:
X = df['weight'].values
y = df['mpg'].values
X = sklearn.utils.shuffle(X,random_state=10)
y = sklearn.utils.shuffle(y,random_state=10)

# defining test size and spliting the data into training and testing:
test_size = 0.2
test_size = int(len(X) * (1-test_size))
X_train = X[:test_size].reshape(-1,1)
X_test = X[test_size:].reshape(-1,1)
y_train = y[:test_size].reshape(-1,1)
y_test = y[test_size:].reshape(-1,1)

# using the classifier:
clf = linreg()
clf.fit(X_train,y_train)

# plotting the data points with best fit line through them:
new_ys = clf.predict(X_test)
plt.scatter(X_test,y_test)
plt.plot(X_test,new_ys,color='r')
plt.show()
# prints the r squared value of line:
print(clf.score(X_test,y_test))
