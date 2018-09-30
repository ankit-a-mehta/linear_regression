
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[66]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error


# In[5]:


train = pd.read_csv('D:\GreyAtom\DataSet_Train/train.csv')
train_copy = train
train.info()


# In[4]:


train.head()


# In[39]:


correlation_values = train.select_dtypes(include=[np.number]).corr()
correlation_values


# In[43]:


#correlation_values[['SalePrice']]
#Remove ID, MSSubClass, OverallCond, 
selected_features = correlation_values[["SalePrice"]][(correlation_values["SalePrice"]>=0.6)|(correlation_values["SalePrice"]<=-0.6)]
selected_features


# In[53]:


a = list(selected_features.index)
train[a].corr()


# In[61]:


reg = LinearRegression()
X = train[['OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageArea']]
y = train[['SalePrice']]


# In[63]:


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)
reg.fit(X_train, y_train)


# In[64]:


y_pred = reg.predict(X_test)


# In[65]:


reg.score(X_test, y_test)


# In[67]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse


# In[70]:


reg = LinearRegression()
X = train[['OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'GarageCars']]
y = train[['SalePrice']]

X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print(reg.score(X_test, y_test))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse


# In[69]:


reg = LinearRegression()
X = train[['OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'GarageCars', '1stFlrSF']]
y = train[['SalePrice']]

X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 42)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print(reg.score(X_test, y_test))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse


# In[44]:


sns.heatmap(correlation_values)

