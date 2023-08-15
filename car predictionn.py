#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler
from sklearn.linear_model import LinearRegression
dataframe=pd.read_csv("CarPrice_Assignment.csv")
dataframe.head()


# In[2]:


dataframe.info()


# In[3]:


dataframe.describe()
dataframe.duplicated().sum()
dataframe.isnull().sum()
dataframe.drop("car_ID", axis=1, inplace=True)
correlation_matrix=dataframe.corr()
correlation_matrix


# In[4]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()


# In[5]:


numerical_cols = []
categorical_cols = []

def get_numerical_and_categorical_columns(dataframe):
    for column in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            numerical_cols.append(column)
        else:
            categorical_cols.append(column)


# In[6]:


plt.figure(figsize=(5,5))
sns.histplot(dataframe["wheelbase"], kde=True)
plt.title("wheelbase", fontsize=14)
plt.xlabel("Wheelbase")
plt.ylabel("Count")
plt.show()


# In[7]:


plt.figure(figsize=(5,5))
sns.kdeplot(data=dataframe, x="carlength", hue=None, multiple="stack")
plt.title("Carlength", fontsize=14)
plt.show()


# In[8]:


sns.histplot(dataframe["carwidth"], kde=True)
plt.title("carwidth", fontsize=14)
plt.xlabel("Carwidth")
plt.ylabel("Count")
plt.show()


# In[9]:


plt.figure(figsize=(5,5))
sns.kdeplot(data=dataframe, x="enginesize", hue=None, multiple="stack")
plt.title("Enginesize", fontsize=14)
plt.show()


# In[13]:


x_train, x_test, y_train, y_test=train_test_split(dataframe.drop("price", axis=1),
                                                  dataframe["price"],
                                                  test_size=0.3,
                                                  random_state=42)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
ohe= OneHotEncoder(handle_unknown="ignore")

x_train_ohe= ohe.fit_transform(x_train[categorical_cols])
x_train_ohe= x_train_ohe.toarray()

x_train_ohe_df= pd.DataFrame(x_train_ohe, columns=ohe.get_feature_names_out([categorical_cols[i]  for i in range(len(categorical_cols))]))

# One-hot encoding removed an index. Let's put it back:
x_train_ohe_df.index= x_train.index

# Joining the tables
x_train = pd.concat([x_train, x_train_ohe_df], axis=1)
x_train.drop(categorical_cols, axis=1, inplace=True)

# Checking result
x_train.head()
x_test_ohe= ohe.transform(x_test[categorical_cols])
x_test_ohe= x_test_ohe.toarray()

x_test_ohe_df= pd.DataFrame(x_test_ohe, columns=ohe.get_feature_names_out([categorical_cols[i] for i in range(len(categorical_cols))]))
#print(x_test_ohe_df)

# One-hot encoding removed an index. Let's put it back:
x_test_ohe_df.index= x_test.index

# Joining the tables
x_test= pd.concat([x_test, x_test_ohe_df], axis=1)

# Dropping old categorical columns
x_test.drop(categorical_cols, axis=1, inplace=True)

# Checking result
x_test.head()
model= LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print (f"model : {model} and  rmse score is : {np.sqrt(mean_squared_error(y_test, y_pred))}, r2 score is {r2_score(y_test, y_pred)}")


# In[ ]:




