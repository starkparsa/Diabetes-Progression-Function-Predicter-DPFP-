#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# The Diabetes Pedigree Function (DPF) is a tool used to assess the genetic risk of diabetes mellitus based on family history. Introduced in 1993 as part of the Diabetes Genetics Initiative, it calculates a numerical score indicating the likelihood of an individual developing diabetes. This score increases with each affected first-degree relative (parents or siblings), with higher scores indicating greater genetic predisposition. The DPF helps stratify individuals into risk categories, although it's important to remember that genetics is just one aspect of diabetes risk, alongside lifestyle and environmental factors. 

# In[94]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Data Visualization
# Let us first load the dataset in a  variable called 'diabetes' using pandas.

# In[95]:


# Loading the diabetes pedigree function dataset
diabetes = pd.read_csv('diabetes.csv')
# Let us drop diabetes outcome such that we can predict diabetes rate accurately
diabetes = diabetes.drop('Outcome',axis = 1)


# Now let us review the contents of the dataframe using `describe()` function and shape of the dataframe using `shape` function to understand the data and start cleaning it.

# In[96]:


diabetes.describe()


# In[97]:


diabetes.shape


# From the above function we can see that there are 768 rows and 8 columns. Here our main focus is on the columns which are:
#    - Pregnancies
#    - Glucose
#    - Blood Pressure
#    - Skin Thickness
#    - Insulin
#    - BMI
#    - Diabetes Pedigree Function(DPF)
#    - Age
# 
# In the folloing we are going to predict Diabetes Pedigree Function(DPF) based on the other columns.

# # Data Pre-processing
# From studying the description of the dataframe we can conclude few things about the data of Prima Indians Diabetes dataset they are:
#  - There are many zeros present in the dataset but zeroes in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` don't make any sense which means they are replacements of null data.
#  - We will replace zero with null using `replace()` function.

# In[98]:


# Loading columns to change to null instead of zero
zero_columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
diabetes[zero_columns] = diabetes[zero_columns].replace(0, np.nan)

diabetes.describe()


# We can now see that the minimum values of `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` are not zeros any more which they are the real ones and null values have been created instead of zeroes to double check it we will use `info()` to check number of non-null values.

# In[99]:


diabetes.info()


# Since there is difference in non-null values we can say we were successful in representing null values. 
# 
# We will do the follwing :
#  - Use simple Imputer to change null value.
#  - Use median startegy because it is not sensitive to outliers.
# 
# Doing this will allow us to make the prediction a bit more accurate.

# In[100]:


from sklearn.impute import SimpleImputer

# Simple imputer function with median strategy
imputer = SimpleImputer(strategy = 'median')

# Imputing the dataset
imputed_diabetes = pd.DataFrame(imputer.fit_transform(diabetes), columns=diabetes.columns)

imputed_diabetes.info()


# After using simple imputer we can see that there are no null values in the dataset. Now let us divide the dataset into X and y where :
#  - X = All columns except DiabetesPedigreeFunction
#  - y = DiabetesPedigreeFunction

# In[101]:


X = imputed_diabetes.drop('DiabetesPedigreeFunction',axis = 1)
y = imputed_diabetes.DiabetesPedigreeFunction


# We will now Split X and y into **X_train**, **X_test** and **y_train**, **y_test** where train values are 70% and test being 30% of original.

# In[102]:


from sklearn.model_selection import train_test_split

# Splitting data
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state= 1)


# # Fitting and Prediction
# 
# We will first use Random Forest to fit and predict values with 50 n_estimators and random_state as 1.

# In[103]:


from sklearn.ensemble import RandomForestRegressor

# RandomForestRegressor variable creation
rfr = RandomForestRegressor(n_estimators= 50, random_state=1)

# Fitting train data using RandomForestRegressor
rfr.fit(X_train,y_train)


# In[104]:


from sklearn.metrics import mean_absolute_error

# Predicting test data
y_pred = rfr.predict(X_test)

# Mean Absolute Error
mae_rfr = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error for Random Forest Regression:", mae_rfr)


# We will now use XG Boost to train and predict values with Learning rate of 0.009 and max leaves of 10.

# In[106]:


from xgboost import XGBRegressor

# XG Boost variable creation
xgbr = XGBRegressor(learning_rate = 0.009,max_leaves = 10)

# Fitting train data using XGBRegressor
xgbr.fit(X_train,y_train)


# In[107]:


y_pred = xgbr.predict(X_test)

mae_xgbr = mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error for XG Boost:", mae_xgbr)


# # Result 
# 
# After comapring both random forest and xg boost we can say that xg boost performed well in this scenario based on the median absolute error(mae). where the mae value of xg boost is 0.24156465896486715. Therefore we will be going forward with XG Boost algorithm for this dataset and problem.

# In[110]:


# Function of the XG Boost
   
def predict(self, Preg, Glu, BP, ST, Insu, BMI, Age):
        X_user = [[Preg, Glu, BP, ST, Insu, BMI, Age]]
        X_user = pd.DataFrame(X_user, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age'])
        y_user = self.model.predict(X_user)
        percent = y_user * 100 / 2.42
        return y_user, percent

