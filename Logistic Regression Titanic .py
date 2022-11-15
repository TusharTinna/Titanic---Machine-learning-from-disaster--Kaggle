#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from sklearn import preprocessing
import matplotlib.pyplot as plt 


# In[2]:


train_df = pd.read_csv("train.csv")

test_df = pd.read_csv("test.csv")

train_df.head()


# In[3]:


# check missing values in train data
train_df.isnull().sum()


# We try to fill the missing values of Age. 
# Cabin has a lot of missing values so it will be good to remove the column from our model.
# Embarked is missing in two values so we will remove the two rows.

# In[4]:


print('The number of samples into the train data is {}.'.format(train_df.shape[0]))


# In[5]:


#Let's see what the 'Age' variable looks like in general.
mean=train_df["Age"].mean()
median=train_df["Age"].median()
a = train_df["Age"].hist(bins=20,density=True  )
train_df["Age"].plot(kind='density')
a.set(xlabel='Age')
plt.axvline(x=mean, color='r', linestyle='-')
plt.axvline(x=median, color='y', linestyle='-')
plt.show()


# As median is closer to the peak we will fill the missing values with the median value

# In[6]:


print(median)


# In[7]:


train_data = train_df.copy()


# In[8]:


train_data.head()


# In[9]:



print('The number of samples into the train data is {}.'.format(train_df.shape[0]))


# In[10]:


train_data = train_data.dropna(subset=['Embarked'])


# In[11]:


train_data["Age"].fillna(median, inplace=True)


# In[12]:


#dropping the cabin 
train_data.drop('Cabin', axis=1, inplace=True)


# In[13]:


# check missing values in adjusted train data
train_data.isnull().sum()


# In[14]:


print('The number of samples into the train data is {}.'.format(train_data.shape[0]))


# SibSp and Parch relate to traveling with family. For simplicity's sake (and to account for possible multicollinearity), I'll combine the effect of these variables into one categorical predictor: whether or not that individual was traveling alone.

# In[15]:


## Create categorical variable for traveling alone
train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)


# In[16]:


#create categorical or dummy variables and drop some variables
training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()


# Now, apply the same changes to the test data.

# In[17]:


test_df.isnull().sum()


# In[18]:


test_data = test_df.copy()
test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[20]:


Selected_features = ['Age', 'Fare','TravelAlone', 'Pclass_1', 'Pclass_2','Pclass_3', 'Embarked_C', 
                     'Embarked_S','Embarked_Q','Sex_male']
X = final_train[Selected_features]
y = final_train['Survived']


# In[21]:


X.head


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[23]:


from sklearn import metrics
reg = LogisticRegression()


# In[24]:


reg.fit(X_train,y_train)


# In[25]:


reg.score(X_train,y_train)


# In[26]:


reg.intercept_


# In[27]:


reg.coef_


# In[28]:


reg.score(X_test,y_test)


# In[31]:


# Model Building
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,VotingRegressor

# Metrics to evaluate the model
from sklearn.metrics import mean_squared_error as mse


# In[32]:


# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
model_dt=DecisionTreeRegressor(criterion='mse')
model_dt.fit(X_train,y_train)


# In[33]:


model_dt.score(X_train,y_train)


# In[34]:


model_dt.score(X_test,y_test)


# In[35]:


# Random forest model
model_rf=RandomForestRegressor(n_estimators=500)
model_rf.fit(X_train,y_train)


# In[36]:


model_rf.score(X_train,y_train)


# In[37]:


model_rf.score(X_test,y_test)


# In[38]:


# Linear Regression
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[39]:


lr.score(X_train,y_train)


# In[40]:


lr.score(X_test,y_test)


# In[41]:


# AdaAdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
model_adb=AdaBoostRegressor(n_estimators=250)
model_adb.fit(X_train,y_train)


# In[42]:


model_adb.score(X_train,y_train)


# In[43]:


model_adb.score(X_test,y_test)


# In[44]:


from sklearn.ensemble import RandomForestClassifier


# In[72]:


rf_clf = RandomForestClassifier(max_depth = 5, random_state=42)
rf_clf.fit(X_train, y_train)


# In[73]:


rf_clf.score(X_train,y_train)


# In[74]:


rf_clf.score(X_test,y_test)


# In[ ]:


prediction = rf_clf.predict(X_test)


# In[75]:


final_test['Survived'] = rf_clf.predict(final_test[Selected_features])
final_test['PassengerId'] = test_df['PassengerId']


# In[76]:


submission = final_test[['PassengerId','Survived']]

submission.to_csv("submissionrcf.csv", index=False)

submission.tail()


# In[ ]:




