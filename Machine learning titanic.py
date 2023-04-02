#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


# In[3]:


# train data
train_data = pd.read_csv("https://raw.githubusercontent.com/rebekz/datascience_course/main/data/titanic/train.csv")

# test data (without labels)
test_data = pd.read_csv("https://raw.githubusercontent.com/rebekz/datascience_course/main/data/titanic/train.csv")


# In[4]:


# show the first 5 rows
train_data.head()


# In[5]:


# data types from the features
train_data.dtypes


# In[6]:


# check how many data points there are 
train_data.shape


# In[7]:


# show how many null values there are in the data set
train_data.isnull().sum()


# In[9]:


# First, let's look at how many women and men were guests on the Titanic:
# Divide the passengers of the Titanic into men and women
women_data = train_data.loc[train_data.Sex == 'female']
men_data = train_data.loc[train_data.Sex == 'male']

fig = px.bar(title="Men and women guest",
             x=["male", "female"],
             y=[len(women_data), len(men_data)],
             labels={
                 "x": "Gender",
                 "y": "Number"
             })
fig.show()


# In[10]:


# Let's take a look at the survival rate of the passengers:
# Show how many of the respective slaughterers have survived
women_kids_survived = women_data.loc[women_data.Age <= 18]["Survived"].sum()
men_kids_survived = men_data.loc[men_data.Age <= 18]["Survived"].sum()

women_survived = women_data["Survived"].sum() - women_kids_survived
men_survived = men_data["Survived"].sum() - men_kids_survived

layout = go.Layout(title="Survival rate between men and women", barmode='stack', xaxis=dict(title='Gender'), yaxis=dict(title='Number of survivors'))
fig = go.Figure(data=[
    go.Bar(name='Adults', x=["male", "female"], y=[men_survived, women_survived], text=[men_survived, women_survived]),
    go.Bar(name='Children', x=["male", "female"], y=[men_kids_survived, women_kids_survived], text=[men_kids_survived, women_kids_survived])
], layout=layout)
fig.show()


# In[11]:


# Let's see how many percent of the passengers died:
# view in percent
fig = px.pie(values=[women_data["Survived"].sum(), len(women_data["Survived"]) - women_data["Survived"].sum()],
             names=["Survived", "Died"],
             color=["Survived", "Died"],
             title="Female survival rate")
fig.show()


# In[12]:


# view in percent
fig = px.pie(values=[men_data["Survived"].sum(), len(men_data["Survived"]) - men_data["Survived"].sum()],
             names=["Survived", "Died"],
             color=["Survived", "Died"],
             title="Male survival rate")
fig.show()


# In[13]:


# Checking the survival rate for different classes
first_class_passangers = train_data.loc[train_data.Pclass == 1]
second_class_passangers = train_data.loc[train_data.Pclass == 2]
thirs_class_passangers = train_data.loc[train_data.Pclass == 3]

first_class_passangers_survived = first_class_passangers["Survived"].sum()
second_class_passangers_survived = second_class_passangers["Survived"].sum()
thirs_class_passangers_survived = thirs_class_passangers["Survived"].sum()

layout = go.Layout(title="Survival rate in classes", barmode='stack', xaxis=dict(title='Class'), yaxis=dict(title='Passangers'))
fig = go.Figure(data=[
    go.Bar(name='Survived', x=[1, 2, 3], y=[first_class_passangers_survived, second_class_passangers_survived, thirs_class_passangers_survived]),
    go.Bar(name='Died', x=[1, 2, 3], y=[len(first_class_passangers) - first_class_passangers_survived, len(second_class_passangers) - second_class_passangers_survived, len(thirs_class_passangers) - thirs_class_passangers_survived])
], layout=layout)
fig.show()


# In[15]:


# calculation of the correlation matrix
sns.clustermap(
    train_data.corr(),
    linewidth=10,
    annot=True,
    annot_kws={"fontsize": 10},
    figsize=(8, 8),
    cbar_pos=(1.04, .2, .03, .4)
)
plt.show()


# In[16]:


# Cabin imputation
# drop the column 'Cabin' from the train data
train_data.drop(['Cabin'], axis=1, inplace=True)


# In[17]:


# Embarked imputation
train_data['Embarked'].value_counts()


# In[18]:


# replace the missing values with the mode
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])


# In[19]:


# Age imputation
women_median = train_data.loc[train_data.Sex == 'female']["Age"].median()
men_median = train_data.loc[train_data.Sex == 'male']["Age"].median()

# replace the missing values with the median of each group (male, female)
train_data.loc[(train_data['Age'].isnull()) & (train_data['Sex'] == "female"), 'Age'] = women_median
train_data.loc[(train_data['Age'].isnull()) & (train_data['Sex'] == "male"), 'Age'] = men_median


# In[20]:


# check if all missing values have been replaced
train_data.isnull().sum()


# In[21]:


# print the first 5 rows of the updated dataframe
train_data.head()


# In[22]:


# show the columns
train_data.columns


# In[23]:


# remove unnecessary features
train_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


# In[24]:


# one-hot encoding for the nominal characteristics
features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked", "Fare"]
X_train = pd.get_dummies(train_data[features])
y_train = train_data["Survived"]


# In[29]:


# print the first 5 rows of the final dataframe
X_train.head()


# In[30]:


# check the test data
test_data.isnull().sum()


# In[31]:


# drop the unnecessary features
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[32]:


# check the test data
test_data.isnull().sum()


# In[33]:


# replace the missing values with the mode
test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].median())


# In[34]:


# Age imputation
women_median = test_data.loc[test_data.Sex == 'female']["Age"].median()
men_median = test_data.loc[test_data.Sex == 'male']["Age"].median()

# replace the missing values with the median of each group (male, female)
test_data.loc[(test_data['Age'].isnull()) & (test_data['Sex'] == "female"), 'Age'] = women_median
test_data.loc[(test_data['Age'].isnull()) & (test_data['Sex'] == "male"), 'Age'] = men_median


# In[35]:


# check the test data
test_data.isnull().sum()


# In[36]:


# one-hot encoding for the nominal characteristics
features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked", "Fare"]
X_test = pd.get_dummies(test_data[features])


# In[37]:


# print the first 5 rows of the final dataframe
X_test.head()


# In[38]:


from sklearn import preprocessing

# MinMaxScaler on the train data
scaler_train = preprocessing.MinMaxScaler()
scaler_train.fit(X_train)

X_train_scaled = scaler_train.transform(X_train)

# MinMaxScaler on the test data
scaler_test = preprocessing.MinMaxScaler()
scaler_test.fit(X_test)

X_test_scaled = scaler_test.transform(X_test)


# In[61]:


# Train: Logistic Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# parameter: LogisticRegression().get_params().keys()
param_grid = {'solver': ['saga', 'liblinear'],
              'penalty': ['l1', 'l2'],
              'C': [0.0001,0.001,0.01,0.1,1,10,100]}

# train model
log_reg_cv = GridSearchCV(LogisticRegression(), cv=10, param_grid=param_grid, scoring='accuracy', n_jobs=-1)
log_reg_cv.fit(X_train_scaled, y_train)

print("Best accuracy: " + str(log_reg_cv.best_score_))
print("Best parameters: " + str(log_reg_cv.best_params_))


# In[62]:


# predict the data from the test data set
predictions = log_reg_cv.predict(X_test_scaled)

# save the predictions in to csv
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('output.csv', index=False)






