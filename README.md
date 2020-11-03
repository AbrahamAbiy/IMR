# IMR
Problem Statement

The State of the World’s Children report by UNICEF provides a rich dataset which depicts a global view of challenges to children’s health, development, and mortality. My aim is to more deeply examine child mortality, which is closely linked to malnutrition and access to health care. Specifically, under-five mortality is one of the principal indicators of progress in child well-being. Additionally, I used another data source to look more closely at pneumonia risk factors and the links to under-5 mortality. Through data exploration, I aim to visualise a global comparison of these statistics, as well as use other data modelling techniques to perform classification of given countries as developed or developing based on their child mortality metrics. I also aim to predict mortality rate given a new country with the relevant given variables. 

Hypothesis

By examining this rich dataset and utilising data analytics techniques, 2 algorithms will be used to make inferences about the data and make predictions. 
I.	Linear Regression: with this initial approach I will attempt to model the child mortality over time given historical results per country. The main aim is the prediction of child mortality statistics given a new country is created.The hypothesis in this case is in the fit of a predictive model given the observed data set.
II.	Classification: by exploring additional research, I can enrich the data by adding a variable related to the development status (Developed/Developing/Least developed) for a subset of the data. Using classification techniques, I aim to classify countries’ development given by the statistics given by the State of the World’s Children Report. 


Dataset

The United Nations Children’s Fund (UNICEF) provides humanitarian and developmental aid to children globally. Some of its crucial activities include immunisations, promoting education and giving emergency relief when disasters occur. In addition to this work, it conducts research to highlight the need for continued development of child healthcare and rights. Crucially, this comes in the form of “The State of the World’s Children '' which is a report with accompanying statistics which are published annually. This data is publically available and will be the focus of our project as it gives a thorough view of child mortality in every country. I also utilised data which detailed risk factors for pneumonia from the Global Burden of Disease Study.

Data Pre-processing

The datasets were not pre processed so some cleaning to remove unnecessary elements were carried out to prepare the data before model implementation.

Linear Regression

First a dataset was obtained which detailed pneumonia risk factors in children for various years ranging from 1990-2017. For my purposes I only required data for 2017, so I removed all other years from the dataset. In addition, two datasets were joined to give a holistic view of child mortality for each country, which I used as the target,  and the related risk factors i.e. features of my dataset.

Logistic Regression

The file was converted from XLSX to Comma Separated Values, this allowed me to identify each column and row as shown below. Extra elements make the dataset harder to analyse e.g. the empty columns in between dates, this was removed to make the dataset more compact. I also removed all empty spaces in the dataset to help with the analysis. 

Secondly there are notes about the dataset which were irrelevant for the algorithms to be implemented, but only informative for the user. This information was removed to ease the analysis for the model.

Another issue with the data is that there were countries with no statistics: Anguilla, British Virgin Islands, Holy See, Liechtenstein, Montserrat, Tokelau. The rows containing these countries haven’t been removed as the model was still able to analyse and make predictions, displaying a message stating that there was not enough data available for these entries.


Data Exploration

After the data was cleaned and pre-processed, the next step was to explore the data. This helped me to  gain a broader picture of potential trends or points to have when doing further analysis on the data.

Linear Regression

In the analysed dataset the variables involved were the country/region and the rate of mortality over different decades of children. My target for the model was the mortality rate of a specific country, and the risk factors were manipulated as predictors. The dataset had a large number of variables which are the different countries and regions as each had its own mortality rate for different age groups, making it more complex. An idea that I developed was the use of a linear regression model which was able to predict the death mortality per country, based on the related variables, where the user inserts the country to be analysed.

In exploring the data, I determined there were no major outliers in terms of the distributed mortality rate. To examine the dataset, I decided to plot scatter graphs for child mortality against each predictor individually. In Graph 1  can see that there is a linear relationship between child mortality and the predictor outdoor air pollution. This process was repeated with all other variables to see the correlation and check for outliers. 
 
Logistic Regression 

From the other dataset I chose had one binary column, I decided to perform logistic regression to classify the dataset where the model is classification where a country and their mortality rate can be classified in 1st/2nd or 3rd world countries by using a benchmark to divide the different categories.
Feature Selection

Feature Selection

Linear Regression

With the linear regression, the predictors were the risk factors and the target was the child mortality. There were originally 12 predictors in the dataset, and for the feature selection I decided to use backward selection to determine the most relevant features. I did this by running the linear regression and then removing the predictor that had the highest p value, this process was repeated until all variables had p values of less than 0.05. The 2 variables I removed were ‘Indoor air pollution from solid fuels’ and ‘Vitamin A deficiency’. After this process I was left with 10 variables which all but had p values of 0.0, which can be seen in Table 1. 
 
Logistic Regression

The goal of the model created was to predict the child mortality death rate of a new country based on the current status of the current countries available for the 2017 year. I considered the negative impact of irrelevant features on model performance, therefore carried out the process of feature selection. The aim was to reduce overfitting - by removing redundant data there was less chance the model would make decisions based on noise. Additionally I aimed for improved accuracy, by reusing misleading data, improving results and lowering training time.
 
The features present in my dataset were the following: 

●	Countries

●	Under-5 mortality rate (deaths per 1,000 live births) (1990/2000/2018)

●	Annual rate of reduction in under-5 mortality rate

●	Under-5 mortality rate by sex (deaths per 1,000 live births) (male vs female 2018)

●	Infant mortality rate (deaths per 1,000 live births) (1990/2018)

●	Neonatal mortality rate (deaths per 1,000 live births) (1990/2000/2018)

The target of the model was an estimation of whether the country is developed or underdeveloped. The feature named Country was eliminated during pre-processing.
The other features will be selected through the feature importance method, which gives a score of each feature, where the higher the score the more relevant this feature is for my input variable.

 
 

I selected the best 4 features:

●	Neonatal mortality rate (deaths per 1,000 live births) 1990

●	Neonatal mortality rate (deaths per 1,000 live births) 2000

●	Neonatal mortality rate (deaths per 1,000 live births) 2018

●	Infant mortality rate (deaths per 1,000 live births) 1990

 

 

To visualize my results my created a 2-d plot of my regression line for each variable individually using the predictions from my model.

Methodology

Linear Regression

A main method utilised in this project was linear regression, the goal was to predict the child mortality of under 5 for a new country given the predictors. As discussed, the dataset used only the year 2017, some features of the selected dataset were unnecessary: ID code, country name, country code and the year. This information was not meaningful for the model where the prediction is needed.

The next step was then to split the dataset into prediction and label; as explained above the goal of this model is to be able to predict the under 5 child mortality of any new country given the predictors. In this case, my label is the under 5 mortality rate while the predictors are all the other features.

 

The next step was to then divide the dataset into the training part and testing part. In my case a 80/20 split was chosen which meant 80% of the data is used to train the model while the other 20% is used to test the model, I then create an object called regressor in the linear regression class, I then fit the linear regression model to the training set. Net the model attempted to predict using the last test set result.

 

 

To visualize my results I created a 2-d plot of my regression line for each variable individually using the predictions from my model.

 
Logistic Regression

For the logistic regression task I aimed to perform a classification of the given countries as to whether they were developed or developing, based on their child mortality metrics. 

 

Firstly I imported the required libraries as seen above. Next I loaded the dataset, defined which columns contained the predictors (X) and the target variable (y), and split the data into test and training sets with 20:80 split as seen below.

 

After this I performed feature selection, utilizing the feature importance method. After determining the most important features I defined a new set of X with only these important features. This can be seen below.

 
Following this I performed the logistic regression making use of the Logit function from the statsmodels.api module. Additionally, I then printed the accuracy of the model using model predictions.

 

Lastly I analyzed the effectiveness of my model by creating a confusion matrix, calculating the precision and recall, and creating a ROC curve as seen below.

Results 

Linear Regression

My linear regression model had an R square value of 1.00, which means that it was able to capture all of the variability in the data. I can therefore conclude that I produced a very good linear regression model. To visualize my results I produced 2-d plots of the least squares prediction of the target variable and the feature in question.

  

To visualize all the results I also produced a plot which shows all of predicted data points on a graph where each of the colours represents a single feature.

The features which had the largest correlation with infant mortality were; 'Child underweight', 'Zinc deficiency' and 'Second-hand smoke'.
 
 
Logistic Regression

My model had an accuracy of 0.92 on the test set which is high.

 
 

For further analysis I also looked at the precision matrix of my model. My model had a very high true positive rate of 151/194 and a true negative rate of 17/194, this is high indicating good performance from my model.


Lastly I produced a ROC curve to visualize the results of my classifier, shown by Figure 5. The AUC of my model 0.791 which means my model was able to distinguish between classes 79.1% of the time.


 
 
 

 

