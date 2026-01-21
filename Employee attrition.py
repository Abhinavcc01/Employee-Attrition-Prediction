#!/usr/bin/env python
# coding: utf-8

# # <span style = "color: green"> Employee attrition Classification </span>

# ***

# The issue of keeping one's employees happy and satisfied is a perennial and age-old challenge. If an employee you have invested so much time and money leaves for "greener pastures", then this would mean that you would have to spend even more time and money to hire somebody else. In the spirit of Kaggle, let us therefore turn to our predictive modelling capabilities and see if we can predict employee attrition on this synthetically generated IBM dataset.

# ### Let's Dive into it

# #### Import necessary libraries

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Read 'HR-Employee-Attrition.csv' dataset and store it inside a variable

# In[6]:


df=pd.read_csv('HR-Employee-Attrition.csv')


# ### Check head

# In[7]:


pd.set_option("display.max_columns", None)


# In[8]:


df.head()


# ### Check last 5 rows

# In[9]:


df.tail()


# ### Check shape

# In[10]:


df.shape


# ### View info about the dataset

# In[11]:


df.info()


# ### View basic statistical information about the dataset

# In[12]:


df.describe()


# ### Check for null values

# In[13]:


df.isna().sum()


# ### View unique values in all categorical columns

# In[14]:


df.columns


# In[15]:


categorical_columns=['Attrition', 'BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18', 'OverTime']
for i in categorical_columns:
    print(f'Unique values in {i} are:',df[i].unique())


# ### Check the number of unique values in all columns

# In[16]:


df.nunique()


# ### Print out the names of the columns having only one unique values 

# In[17]:


columns_one_unique=[col for col in df.columns if df[col].nunique()==1]
for col_name in columns_one_unique:
    print(f'{col_name}')


# ### Drop these columns as they won't be useful in our predicition

# In[18]:


df=df.drop(columns=['EmployeeCount','Over18','StandardHours'])
df.head()


# ### Drop EmployeeNumber column aswell

# In[19]:


df=df.drop(columns=['EmployeeNumber'])
df.head()


# ### Create following groupby valuecounts

# In[20]:


df.groupby(['Department','EducationField','Gender']).size()


# # <span style = "color: orange"> Data Visualization </span>

# ### Plot the following

# In[21]:


plt.figure(figsize=(4,3))
sns.countplot(x='Attrition',data=df,palette=['blue','orange'])
plt.xlabel('Attrition')
plt.ylabel('Number of employees')
plt.show()


# In[22]:


plt.figure(figsize=(4,3))
sns.countplot(x='Attrition',data=df,hue='Gender')
plt.xlabel('Attrition')
plt.ylabel('Number of employees')
plt.show()


# In[23]:


sns.countplot(x='BusinessTravel',data=df,hue='JobRole')
plt.show()


# In[24]:


sns.countplot(x='OverTime',data=df,hue='MaritalStatus')
plt.show()


# In[25]:


stats=df.groupby('JobRole')['MonthlyIncome'].agg(['mean','std'])
colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'brown','pink','olive','magenta',]
plt.barh(stats.index,stats['mean'],xerr=stats['std'],color=colors)
plt.xlabel('MonthlyIncome')
plt.ylabel('JobRole')
plt.show()


# In[26]:


plt.figure(figsize=(6,4))
stats = df.groupby('Gender')['MonthlyIncome'].agg(['mean', 'std']).sort_values('mean')
colors=['blue','orange']
plt.barh(stats.index,stats['mean'],xerr=stats['std'],color=colors)
plt.xlabel('MonthlyIncome')
plt.ylabel('Gender')
plt.show()


# In[27]:


stats = df.groupby('EducationField')['MonthlyIncome'].agg(['mean', 'std']).sort_values('mean')

plt.figure(figsize=(6, 4))
colors = ["skyblue", "orange", "green", "red", "purple", "brown"]
plt.barh(stats.index,stats['mean'], xerr=stats['std'],color=colors)

plt.xlabel('MonthlyIncome')
plt.ylabel('EducationField')
plt.tight_layout()
plt.show()


# In[28]:


plt.figure(figsize=(10,6))
sns.countplot(x='JobSatisfaction',data=df,hue='JobRole')
plt.show()


# In[29]:


plt.figure(figsize=(4,4))
sns.histplot(x='Age',data=df)
plt.show()


# ### Data Preprocessing

# #### Convert Attrition from ('Yes', 'No') to (1,0) 

# In[30]:


def attrition(value):
    if value=='Yes':
        return 1
    else:
        return 0


# In[31]:


df['Attrition']=df['Attrition'].apply(attrition)
df.head()


# ### Convert the rest of the categorical values into numeric using dummy variables and store the results in a new dataframe called 'newdf'

# In[32]:


categorical_columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus', 'OverTime']
newdf=pd.get_dummies(df,columns=categorical_columns)
newdf.head()


# #### Check the shape of our new dataset

# In[33]:


newdf.shape


# #### Print unique values in our new dataframe

# In[34]:


columns=['Age', 'Attrition', 'DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
       'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']
for i in columns:
    print(f'Unique values in {i} are:',newdf[i].unique())


# #### Split the columns into input and target variables (X and y)

# In[35]:


X=newdf.drop(columns=['Attrition'])
y=newdf['Attrition']


# #### Carry out Feature scaling using StandardScaler

# In[36]:


from sklearn.preprocessing import StandardScaler


# In[37]:


scaler=StandardScaler()


# In[38]:


X_scaled=scaler.fit_transform(X)


# In[39]:


scaled_X=pd.DataFrame(X_scaled,columns=X.columns)


# In[40]:


scaled_X.head()


# ### Split the dataset into training and testing set

# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# ## Machine Learning Models

# ### Logistic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score


# In[44]:


model=LogisticRegression()


# In[45]:


model.fit(X_train,y_train)


# In[46]:


y_pred=model.predict(X_test)


# In[47]:


print('Acuuracy score of our model is:',accuracy_score(y_test,y_pred))
print('Confusion matrix is:',confusion_matrix(y_test,y_pred))
print('Classification report is:',classification_report(y_test,y_pred))
print('Cross validation score is:',cross_val_score(model,X,y,scoring='accuracy').mean())


# <span style = "color:orange"> Visualize confusion matrix </span>

# In[48]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d')
plt.show()


# ### Random Forest Classifier
# ** Choose the best estimator and parameters :GridSearchCV**

# In[49]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[50]:


param_grid={'max_depth': [0.5, 1, 5, 10], 'n_estimators': [16, 32, 50, 100], 'n_jobs': [1, 2],'random_state': [1, 10, 20, 42]}


# In[51]:


gsv=GridSearchCV(RandomForestClassifier(),param_grid,scoring='accuracy')


# In[52]:


gsv.fit(X_train,y_train)


# In[53]:


print('Best estimator is:',gsv.best_estimator_)
print('Best parameters are:',gsv.best_params_)
print('Best score is :',gsv.best_score_)


# <span style = "color:blue"> Create Random forest model with the best parameters </span>

# In[54]:


best_rf_model=gsv.best_estimator_
best_rf_model


# In[55]:


y_pred=best_rf_model.predict(X_test)


# In[56]:


print('Accuracy score is:',accuracy_score(y_test,y_pred))
print('Confusion matrix:',confusion_matrix(y_test,y_pred))
print('Classification report:',classification_report(y_test,y_pred))
print('Cross val score:',cross_val_score(best_rf_model,X,y,scoring='accuracy').mean())


# <span style = "color:orange"> Visualize confusion matrix </span>

# In[57]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d')
plt.show()


# ### Support Vector Machine

# In[58]:


from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10]}

gsv = GridSearchCV(LinearSVC(max_iter=10000),param_grid,cv=3,n_jobs=-1,
    verbose=2
)

gsv.fit(X_train, y_train)


# In[59]:


print('Best estimator is:',gsv.best_estimator_)
print('Best parameters are:',gsv.best_params_)
print('Best score is:',gsv.best_score_)


# In[60]:


best_svc=gsv.best_estimator_
best_svc


# In[61]:


y_pred=best_svc.predict(X_test)


# In[62]:


print('Accuracy score is:',accuracy_score(y_test,y_pred))
print('Confusion matrix :',confusion_matrix(y_test,y_pred))
print('Classification report:',classification_report(y_test,y_pred))
print('Cross val score is:',cross_val_score(best_svc,X,y,scoring='accuracy').mean())


# <span style = "color:orange"> Visualize confusion matrix </span>

# In[63]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d')
plt.show()


# ### AdaBoost: Classifier

# In[64]:


from sklearn.ensemble import AdaBoostClassifier


# In[65]:


model=AdaBoostClassifier(n_estimators=50,learning_rate=1,random_state=100)


# In[66]:


model.fit(X_train,y_train)


# In[67]:


y_pred=model.predict(X_test)


# In[68]:


print('Accuracy score:',accuracy_score(y_test,y_pred))
print('Confusion matrix:',confusion_matrix(y_test,y_pred))
print('Classification report:',classification_report(y_test,y_pred))
print('cross val report:',cross_val_score(model,X,y,scoring='accuracy').mean())


# <span style = "color:orange"> Visualize confusion matrix </span>

# In[69]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d')
plt.show()


# ### View score of different models in one dataframe

# In[70]:


data={'Models':['Gradient Boost Classifier','Logistic Regression','Support Vector Classifier','Random Forest Classifier'],
     'Score':[0.8795,0.8408,0.8707,0.8632]}
data1=pd.DataFrame(data,index=[0,1,2,3])
data1


# ## Use PCA to reduce dimensionality of the data

# #### Import PCA and fit our X_train

# In[71]:


from sklearn.decomposition import PCA


# In[72]:


pca=PCA(n_components=0.95)


# In[73]:


pca.n_components


# #### Apply the mapping (transform) to both the training set and the test set.

# In[74]:


train_X = pca.transform(X_train)
test_X = pca.transform(X_test)


# #### Import machine learning model of our choice, we are going with RandomForest for this problem

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# #### Create RandomForest model with the best parameter we got earlier and train it

# In[ ]:


RandomForestCls=best_rf_model


# In[ ]:


RandomForestCls.fit(train_X,y_train)


# #### Check the score of our model

# In[ ]:


RandomForestCls.score(train_X,y_train)


# #### Make predictions with X_test and check the accuracy score

# In[ ]:


pred=RandomForestCls.predict(test_X)


# In[ ]:


print('Accuracy score is:',accuracy_score(y_test,pred))


# ### Print Confusion matrix and Classification report

# In[ ]:


print('Confusion matrix:',confusion_matrix(y_test,pred))
print('Classification report :',classification_report(y_test,pred))


# # <span style = "color:green"> Good Job! You have Successfully completed one Capstone Project </span>

# ***
