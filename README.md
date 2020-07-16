**Introduction**

  **Random forest** is a supervised learning algorithm. It can be used both for classification and regression. It is also the most flexible and easy to use algorithm. A forest is comprised of trees. It is said that the more trees it has, the more robust a forest is. Random forests creates decision trees on randomly selected data samples, gets prediction from each tree and selects the best solution by means of voting. It also provides a pretty good indicator of the feature importance.
   Tree models where the target variable can take a discrete set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. A random forest is a collection of decision trees. With that said, random forests are a strong modeling technique and much more robust than a single decision tree.

-Random forest uses gini importance or mean decrease in impurity (MDI) to calculate the importance of each feature. 

-Gini importance is also known as the total decrease in node impurity. This is how much the model fit or accuracy decreases when you drop a variable. 

-The larger the decrease, the more significant the variable is. 

-Here, the mean decrease is a significant parameter for variable selection. 

-The Gini index can describe the overall explanatory power of the variables.
 
![0_YEwFetXQGPB8aDFV](https://user-images.githubusercontent.com/66662946/87627957-879f6180-c74d-11ea-9e62-5ccca392d2b1.png)
 
**Predicting Wine Quality**
 
  For this project, I used Kaggle’s https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009 dataset to build  classification model to predict whether a particular sample of red wine is “good quality” or not. Each sample of wine in this dataset is given a “quality” score between 0 and 10. For the purpose of this project, I converted the output to a binary output where each wine is either “good quality” (a score of 7 or higher) or not. The quality of a wine is determined by 11 input variables:
  
 1.Fixed acidity
 
 2.Volatile acidity
 
 3.Citric acid
 
 4.Residual sugar
 
 5.Chlorides
 
 6.Free sulfur dioxide
 
 7.Total sulfur dioxide
 
 8.Density
 
 9.pH
 
 10.Sulfates
 
 11.Alcohol
 
 **Let's get into the implementation.**
 
 Let's determine the quality of given samples of red wine using Random Forest method since it obtain highest precision.
 
 Importing Libraries
 ```
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
%matplotlib inline
```

To read a given red wine dataset:
```
red_wine=pd.read_csv('winequality-red.csv')
```

We can display the total number of rows and columns in the dataset
```
print("Rows,Columns",red_wine.shape)
```
```
1599,12
```

where the first parameter displays the number of rows followed by number of columns.

To obtain first five rows of the data
```
red_wine.head()
```
![30](https://user-images.githubusercontent.com/66662946/87628552-d3063f80-c74e-11ea-95cf-3a248bf83421.png)

Checking the quality based on pH value
```
sns.countplot(x="quality",hue="pH",data=red_wine)
```

To see the correlation between the variables present, we use the correlation matrix
```
corr = red_wine.corr()
plt.pyplot.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
```

Plotting the histogram based on the quality
```
red_wine["quality"].plot.hist()
```

**Converting to classification problem**

To convert the following dataset into classification problem, we must check for the presence of null values.This returns the binary value (true/false).
```
red_wine.isnull()
```

To check the sum of null values present:
```
red_wine.isnull().sum()
```
![40](https://user-images.githubusercontent.com/66662946/87628714-309a8c00-c74f-11ea-9e80-b1af2c8a5bb6.png)

It will be easier to solve the given dataset in the presence of zero null values.


To find the effectiveness of the problem,let us initialize the criteria as 'goodquality' where a particular wine will be effective if its quality is greater than or equal to 7
```
red_wine['goodquality'] = [1 if x >= 7
                            else 0 for x in red_wine['quality']]
```

Let's separate the dependent and independent variable into dataframe, where the independent variable will be the final value for predicting the precision.
```
X = red_wine.drop(['quality','goodquality'], axis = 1)
y = red_wine['goodquality']
```

To check the number of sample of good wine present.
```
red_wine['goodquality'].value_counts()

0    1382
1     217
Name: goodquality, dtype: int64
```
It indicates that out of 1599 wine samples, 217 samples are of good quality. 

Standardising the particular variable(X)
```
from sklearn.preprocessing import StandardScaler
X_features = X
X = StandardScaler().fit_transform(X)
```
Next we have to scale the train and test the data to preprocess the data
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
```

**Modelling the data**

Let's use Random Forest Model to obtain highest precision.

Importing libraries
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
```

Assigning a the Random forest classifier to an variable
```
rfc_model = RandomForestClassifier(random_state=1)
rfc_model.fit(X_train, y_train)
```

**Prediction**
```
pred2 = rfc_model.predict(X_test)
print(classification_report(y_test,pred2))
```

![50](https://user-images.githubusercontent.com/66662946/87634855-b7556600-c75b-11ea-8bc8-3c66983cd175.png)


**Confusion Matrix**

A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.
```
print(confusion_matrix(y_test,pred2))
```

```
[[345  10]
 [ 16  29]]
 ```

**Finding important features**

We are finding important features or selecting features in the Red Wine dataset.
Train the model using train set and find prediction on the train set.
```
from sklearn import ensemble
gbc = ensemble.RandomForestClassifier()
gbc.fit(X, y)
```

We can graph the quantity of the variables present in good quality wine.
```
feat_importances = pd.Series(rfc_model.feature_importances_, index=X_features.columns)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
```

![1](https://user-images.githubusercontent.com/66662946/87627791-24153400-c74d-11ea-8315-487da74a8057.png)

From the above graph we can come to conclusion that a sample of red wine is of good quality if it contains large amount of Alcohol.


Therefore, **Random forests** is considered as a highly accurate and robust method because of the number of decision trees participating in the process.
We can get the relative feature importance, which helps in selecting the most contributing features for the classifier.

