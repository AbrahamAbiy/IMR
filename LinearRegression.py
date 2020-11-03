import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.impute import SimpleImputer
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
%matplotlib

dataset = pd.read_csv('pn_predictors.csv')

#print(dataset.describe())
#print(dataset.isnull().any())

#replace missing values with numpy default
#imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
#Define features
X = dataset[['Outdoor air pollution', 'Child underweight', 'Indoor air pollution from solid fuels', 'No access to handwashing facility', 'Secondhand smoke', 'Vitamin A deficiency', 'Zinc deficiency', 'Short gestation for birth weight', 'Non-exclusive breastfeeding', 'Low birth weight for gestation', 'Child wasting', 'Child stunting']].values

Y = dataset[['Under-5s Death (Number)']].values

# data exploration, check average deaths
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['Under-5s Death (Number)'])

#Split 80% training, 20% testing
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Train model
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)

#find optimal weights
#coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns = ['Coefficient'])
#coeff_df = coeff_df.loc[~coeff_df.index.duplicated(keep='first')]
#coeff_df = pd.DataFrame(regressor.coef_, ['Outdoor air pollution', 'Child underweight', 'Indoor air pollution from solid fuels', 'No access to handwashing facility', 'Secondhand smoke', 'Vitamin A deficiency', 'Zinc deficiency', 'Short gestation for birth weight', 'Non-exclusive breastfeeding', 'Low birth weight for gestation', 'Child wasting', 'Child stunting'], columns=['Coefficient'])
#coeff_df

#Test model
#y_pred = regressor.predict(X_test)

#Actual values V prediction of model
#df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#df1.plot(kind='bar',figsize=(10,8))
#plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()

#Error of model
#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

