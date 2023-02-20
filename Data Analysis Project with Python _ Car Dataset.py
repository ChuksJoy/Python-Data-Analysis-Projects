#!/usr/bin/env python
# coding: utf-8

# # Car Dataset Analysis
# 
# For this project, I will be using a Car Dataset which contains data of different cars with their specifications and this data will be analysed with the Pandas dataframe.
# 
# This is a Data analytics Python project I got from Youtube and you can view the video [here](youtube.com/watch?v=fhiUl7f5DnI&list=PLy3lFw0OTlutzXFVwttrtaRGEEyLEdnpy&index=3)

# In[1]:


## Importing the dataframe

import pandas as pd


# In[2]:


# geting the data
car = pd.read_csv('Cars Data.csv')


# In[3]:


#exploring the dataset

car.head()


# In[4]:


car.shape


# In[5]:


car.dtypes


# In[6]:


car.index


# In[7]:


car.columns


# In[8]:


car.nunique()


# In[9]:


car.count()


# In[10]:


car.info()


# ## Cleaning the dataset
# 
# * checking for Null values
# * filling the nulls with Mean of the column

# In[15]:


car.isnull().sum()


# In[12]:


car = car.fillna(car.mean())


# ### Value Count
# 
# Checking the dataset to see the count of different types of car makes that are in our dataset.

# In[16]:


car.head(2)


# In[17]:


car['Make'].value_counts()


# #### Filtering
# 
# Show all the records where Origin is Asia or Europe

# In[18]:


car.head(2)


# In[19]:


car[car['Origin'].isin(['Asia', 'Europe'])]


# #### Removing Unwanted Records/Rows
# 
# Remove all the records/rows where Weight is above 4000

# In[23]:


car[~(car['Weight'] > 4000)]


# #### Applying Function on a Column
# 
# Increase all the values of MPG_City column by 3

# In[24]:


car.head(3)


# In[25]:


car['MPG_City'] = car['MPG_City'].apply(lambda x:x+3)


# In[26]:


car

