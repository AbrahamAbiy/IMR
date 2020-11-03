#  Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  Importing the dataset:
dataset = pd.read_csv('pn_predictors.csv')
# Y: dependent variable vector
# In the first run X’s type is object due to the different types of #independent variables.State column contains categorical variables
X = dataset.iloc[:, 4:-1].values
Y = dataset.iloc[:, 16].values

plt.scatter(dataset['Outdoor air pollution'], dataset['Under-5s Death (Number)'], color='green')
plt.title('Outdoor air pollution V Child Mortality', fontsize=14)
plt.xlabel('Outdoor air pollution', fontsize=14)
plt.ylabel('Child Mortality', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(dataset['Child underweight'], dataset['Under-5s Death (Number)'], color='green')
plt.title('Outdoor air pollution V Child Mortality', fontsize=14)
plt.xlabel('Child underweight', fontsize=14)
plt.ylabel('Child Mortality', fontsize=14)
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#  Fit multiple Linear Regression model to our Train set
from sklearn.linear_model import LinearRegression
p = Y_test.shape
c = X_test.shape
print(p)
print(c)
# Create an object called regressor in the LinearRegression class…
regressor = LinearRegression()
# Fit the linear regression model to the training set… We use the fit #method the arguments of the fit method will be training sets

regressor.fit(X_train, Y_train)
#  Predicting the Test set results:
y_pred = regressor.predict(X_test)

# Beta0 has x0=1. Add a column of for the the first term of the #MultiLinear Regression equation.
import statsmodels.api as sm
X= np.append(arr = np.ones((188,1)).astype(int), values = X, axis=1)
X_opt= X[:, [0,1,2,3,5,6,8,9,10,11,12]] #Backwards elimination,removed x4 and x7 as p>0.05
#Optimal X contains the highly impacted independent variables
#OLS: Oridnary Least Square Class. endog is the dependent variable, #exog is the number of observations
regressor_OLS=sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())

#predictions
x1n = np.linspace(0,400, 150)
Xnew = np.column_stack((x1n, np.sin(x1n), (x1n-5)**2, np.cos(x1n), 2**x1n, np.tan(x1n), x1n-1, 3**x1n, 4**x1n, x1n, x1n))
Xnew = sm.add_constant(Xnew)
ynewpred =  regressor.predict(Xnew) # predict out of sample
print(ynewpred)

#plot predictions along with Data and true values
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
#ax.plot(X, Y, 'o', label="Data")
#ax.plot(X_test, Y_test, 'b-', label="True")
ax.plot(np.hstack((X_train, Xnew)), np.hstack((Y_train, ynewpred)), 'r', label="OLS prediction")
ax.legend(loc="best");

