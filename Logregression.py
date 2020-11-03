#Import libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#Load dataset
data = pd.read_csv('LogRegDataClean.csv', encoding= 'unicode_escape')
data = data.dropna()


#SMOTE
X = data.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
y = data.loc[:, data.columns == 'Developed/developing']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['Developed/developing'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of developing in oversampled data",len(os_data_y[os_data_y['Developed/developing']==0]))
print("Number of developed",len(os_data_y[os_data_y['Developed/developing']==1]))
print("Proportion of developing data in oversampled data is ",len(os_data_y[os_data_y['Developed/developing']==0])/len(os_data_X))
print("Proportion of developed data in oversampled data is ",len(os_data_y[os_data_y['Developed/developing']==1])/len(os_data_X))


# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# load the iris datasets
#dataset = datasets.load_iris()
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

#input selected features
#cols=['Under-5 mortality rate by sex (deaths per 1,000 live births) 2018 male', 'Neonatal mortality rate (deaths per 1,000 live births) 2000', 'Neonatal mortality rate (deaths per 1,000 live births) 1990', 'Neonatal mortality rate (deaths per 1,000 live births) 2018']
X_opt= data.iloc[:, [5,9,10,11]]




print(y)
print(X_test)

import statsmodels.api as sm
logit_model=sm.Logit(y,X_opt)
result=logit_model.fit()
print(result.summary2())


L = LogisticRegression().fit(X_train, y_train)
predictions = result.predict(X_opt)
# Use score method to get accuracy of model
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(L.score(X_test, y_test)))

#Confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y.round(), predictions.round())

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(L.score(X_test, y_test))
plt.title(all_sample_title, size = 15);

#precision recall
from sklearn.metrics import classification_report
print(classification_report(y.round(), predictions.round()))

#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, L.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, L.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()