# In this script I created my own custom Linear Regression model

# Unfortunately, this linear regression model is limited because if 'data_x' and
# 'data_y' are different numpy shapes, an error will occur, therfore to use this
# model, 'data_x' and 'data_y' must be the same shape.

import numpy as np
import matplotlib.pyplot as plt
import sklearn

class linreg():

    # function takes numpy arrays as parameters:

    def fit(self,data_x,data_y):
        # finding the best fit line:

        # finding the slope of best fit line:
        m_numerator = ( (data_x.mean() * data_y.mean()) - (data_x * data_y).mean() )
        m_denominator = ( (data_x.mean() **2) - (data_x**2).mean() )
        m = m_numerator / m_denominator

        # finding the y-int of best fit line:
        b = data_y.mean() - (m*data_x.mean())

        # creates the Linear Regression model which predicts y values:
        best_fit_line_ys = m*data_x + b

        self.m = m
        self.b = b

    def predict(self,X):
        return self.m * X + self.b


    def score(self,X_test,y_test):

        # finds the r squared value of the Linear Model in which is "the percent of the variablity in y explained
        # by the model:

        predicted_ys = self.m * X_test + self.b
        squared_error_yhat = (y_test - predicted_ys) **2
        squared_error_ymean = (y_test.mean() - predicted_ys) **2
        result = 1 - (squared_error_yhat.sum() / squared_error_ymean.sum())
        return result
