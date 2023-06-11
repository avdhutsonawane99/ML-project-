#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib as plt
import numpy as np


# In[2]:


df=pd.read_csv('data.csv')
df


# In[3]:


df.head()


# In[4]:


df.info()


# # Preprocessing for filling missing attributes
# 

# In[5]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')    
imputer.fit(df)                            


# In[6]:


imputer.statistics_                   


# In[7]:


df_processed=pd.DataFrame(imputer.transform(df),columns=df.columns)   #missing attributes filling


# In[8]:


df_processed                                                  ## processed data with no empty space


# In[9]:


df_processed['CHAS'].value_counts()


# In[10]:


df_processed.describe()


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


df_processed.hist(bins=60, figsize=(25,20))                     # histogram 


# both train and test has same % of CHAS 
# 

# # looking for correlation
# looking at the correlation you will get to know which attribute affect the most to the price so you can plot the histogram to see the effect by selecting those attributes only
# 
# positive effect-increases the price
# 
# negative effect decreases the price 

# In[13]:


corr_matrix= df_processed.corr()               # we can check which attribute affect the most on the output by correlation
#print(corr_matrix)
corr_matrix['MEDV'].sort_values(ascending=False)


# In[14]:


from pandas.plotting import scatter_matrix              # plotting of output and attributes to check the effect of the attributes on output
attributes = ['MEDV','RM','PTRATIO','LSTAT']     
scatter_matrix(df_processed[attributes],figsize=(20,15))


# # Feature Engineering
# ###### Contribution of attributes with high correlation

# In[15]:


#df_processed['TAXRM'] =df_processed['TAX']/df_processed['RM']    


# In[16]:


#df_processed.head()


# In[17]:



#corr_mattrix=df_processed.corr()            # correlation checking of attribute combination
#corr_mattrix['MEDV'].sort_values(ascending=False)


# In[18]:


#df_processed.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# # train test split
#  during train test split there is chance of splitting same kind of value to train set and other value to test set. so we can do stratifiedshufflesplit so values get splitted in equal ratio in both sets

# In[19]:


df_processed['CHAS'].value_counts()    


# In[20]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df_processed,test_size=0.2, random_state=42)
print(f"rows in training set are {len (train_set)}\n and testing set are {len (test_set)}")


# In[21]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train,test in split.split(df , df_processed['CHAS']):
    strat_train_set = df_processed.loc[train]
    strat_test_set = df_processed.loc[test]


# In[22]:


strat_test_set


# In[23]:


strat_test_set.shape


# In[24]:


strat_test_set['CHAS'].value_counts()


# # separating attributes and labels

# In[25]:


attributes_train_set= strat_train_set.drop("MEDV", axis=1)
labels_train_set= strat_train_set['MEDV'].copy()
print (attributes_train_set.shape)
print(labels_train_set.shape)


# # pipeline
# 

# In[26]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ("std_scaler",StandardScaler()),      # you can add as many things as you want
                       ])


# In[27]:


housing = my_pipeline.fit_transform(attributes_train_set)


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing,labels_train_set )


# # Evaluating the model

# In[29]:


from sklearn.metrics import mean_squared_error
prediction = model.predict(housing)
mse = mean_squared_error(labels_train_set, prediction)
rmse = np.sqrt(mse)


# In[30]:


rmse


# # using better evaluation technique 

# In[31]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing, labels_train_set, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[32]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[33]:


print_scores(rmse_scores)


# # Saving the model 

# In[34]:


arg=(model,my_pipeline)
from joblib import dump, load
dump(arg, 'Dragon.joblib') 


# In[35]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[36]:


final_rmse


# In[37]:


from joblib import dump, load
import numpy as np
model,pipe = load('Dragon.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)


# In[38]:


a=np.array([[4.87141,0.0,18.10,0,0.614,6.484,93.6,2.3053,24,666,20.2,396.21,18.68]])
X_test_prepared = my_pipeline.transform(a)
print(X_test_prepared)
final_predictions = model.predict(X_test_prepared)
print(final_predictions)

