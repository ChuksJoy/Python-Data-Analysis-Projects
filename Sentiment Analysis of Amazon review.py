#!/usr/bin/env python
# coding: utf-8

# ### Sentiment Analysis 
# 
# For this project, I will be implementing Sentiment analysis of amazon customer reviews using Python.
# 
# 

# In[1]:


# importing our libraries


import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re #regular expression
from textblob import TextBlob # used to process textual data
from wordcloud import WordCloud # an image if words and use cloud creator to highlight popular words and phrases
import seaborn as sns # for dat exploration and visualization
import matplotlib.pyplot as plt #cross platform data viz package 
import cufflinks as cf #links plotly with pandas so that charts can be easily created
get_ipython().run_line_magic('matplotlib', 'inline')
#cause the plots and graphs to appear below the cell where the plotting command are entered and offers backend activities to frontend
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
cf.go_offline();
# in order to display the plot inside the notbook, the above three lines are used to initiate the plotly notebook mode.
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')
warnings.warn('this will not show')

pd.set_option('display.max_columns', None)


# In[2]:


# imoorting our dataset
df = pd.read_csv('amazon data.csv')


# In[3]:


df.head()


# In[4]:


df.describe


# In[5]:


df.columns


# In[6]:


# sorting the wilson_lower_bound into descending order

df = df.sort_values('wilson_lower_bound', ascending = False)


# In[7]:


# drop the unnamed column
df.drop('Unnamed: 0', inplace = True, axis=1)


# In[8]:


df.head(2)


# In[9]:


# checking for missing values
df.isnull().sum()


# In[10]:


# making a function for missing values
def missing_values_analysis(df):
    na_columns_ = [col for col in df.columns if df[col].isnull().sum()> 0]
    n_miss = df[na_columns_].isnull().sum().sort_values(ascending = True)
    ratio_ = (df[na_columns_].isnull().sum()/df.shape[0]* 100).sort_values(ascending = True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis =1, keys=['Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

#another function for checking dataframe
def check_dataframe(df, head=5, tail = 5):
    
    print('SHAPE'.center(82, '~'))
    print('Rows: {}'.format(df.shape[0]))
    print('columns: {}'.format(df.shape[1]))
    print('TYPES'.center(82, '~'))
    print(df.dtypes)
    print("".center(82, '~'))
    print(missing_values_analysis(df))
    print('DUPLICATED VALUES'.center(82, '~'))
    print(df.duplicated().sum())
    print('QUANTILES'.center(82, '~'))
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    
check_dataframe(df)


# The code above defines two functions, "missing_values_analysis" and "check_dataframe", and then calls the "check_dataframe" function with an argument "df".
# 
# The "missing_values_analysis" function takes a DataFrame as input and returns a new DataFrame that shows the number and percentage of missing values for each column in the input DataFrame.
# 
# The "check_dataframe" function takes a DataFrame as input and prints out several pieces of information about the DataFrame, including its shape, data types, missing values, number of duplicated values, and quantiles.

# In[11]:


# writing a function for checking unique values in each columns
def check_class(dataframe):
    nunique_df = pd.DataFrame({'Variable': dataframe.columns,
                              'Classes': [dataframe[i].nunique()\
                                         for i in dataframe.columns]})
    nunique_df = nunique_df.sort_values('Classes', ascending = False)
    nunique_df = nunique_df.reset_index(drop = True)
    return nunique_df

check_class(df)


# The function first creates a new DataFrame called "nunique_df" that has two columns: "Variable" and "classes". The "Variable" column contains the names of the columns in the input DataFrame, and the "classes" column contains the number of unique values in each column.
# 
# The function then sorts the "nunique_df" DataFrame by the "classes" column in descending order and resets the index to start at 0.
# 
# The function returns the sorted and indexed "nunique_df" DataFrame.

# #### Performing an Overall Categorical Variable analysis
# 
# Writing a Python function called categorical_variable_summary that takes in a Pandas DataFrame and the name of a categorical variable column, and creates a subplot consisting of a count plot and a pie chart for that column using Plotly, with a specific set of color constraints

# In[12]:


constraints = ('#FF0000', '#800080', '#FFFF00', '#A52A2A', '#FFA500')

def categorical_variable_summary(df, column_name):
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Countplot', 'Percentage'),
                       specs=[[{"type": "xy"},{'type':'domain'}]])

    fig.add_trace(go.Bar(y=df[column_name].value_counts().values.tolist(),
                        x=[str(i) for i in df[column_name].value_counts().index],
                        text=df[column_name].value_counts().values.tolist(),
                        textfont=dict(size=14),
                        name=column_name,
                        textposition='auto',
                        showlegend=False,
                        marker=dict(color=constraints,
                                    line=dict(color='#DBE6EC',
                                               width=1))),
                 row=1, col=1)

    fig.add_trace(go.Pie(labels=df[column_name].value_counts().keys(),
                        values=df[column_name].value_counts().values,
                        textfont=dict(size=18),
                        textposition='auto',
                        showlegend=False,
                        name=column_name,
                        marker=dict(colors=constraints)),
                 row=1, col=2)
    
    fig.update_layout(title={'text': column_name,
                            'y': 0.9,
                            'x': 0.5,
                            'xanchor':'center',
                            'yanchor':'top'},
                     template='plotly_white')
    iplot(fig)


# The categorical_variable_summary function takes a DataFrame df and a column name column_name as inputs. It creates a subplot consisting of a count plot and a pie chart, each showing the distribution of values in the specified column. The function uses the make_subplots function from the plotly.subplots module to create the subplot, and the go.Bar and go.Pie classes from the plotly.graph_objects module to create the count plot and pie chart, respectively.
# 
# The count plot is created by adding a go.Bar trace to the subplot. The y attribute of the trace is set to the count of each unique value in the specified column, and the x attribute is set to the unique values themselves (converted to strings). The text attribute is set to the count of each value (as a string), which is displayed on the bar. The marker attribute is set to a dictionary with a color key whose value is set to the constraints tuple, which specifies the color for each bar. The row and col attributes are set to 1 and 1, respectively, indicating that the trace should be added to the first subplot.
# 
# The pie chart is created in a similar way, by adding a go.Pie trace to the subplot. The labels and values attributes of the trace are set to the unique values and their counts, respectively, in the specified column. The marker attribute is set to a dictionary with a colors key whose value is set to the constraints tuple, which specifies the color for each wedge of the pie. The row and col attributes are set to 1 and 2, respectively, indicating that the trace should be added to the second subplot.
# 
# The layout of the subplot is customized using the update_layout method of the fig object. The title attribute is set to a dictionary with a text key whose value is set
# 
# 
# 
# 

# In[13]:


categorical_variable_summary(df, 'overall')


# ##### Data Cleaning 

# In[14]:


df.reviewText.head()


# In[15]:


review_example = df.reviewText[2023]
review_example


# In[42]:


review_example = df.reviewText[2031]
review_example


# In[43]:


#after seeing the data we will clean it from punctuation using regex, that is removing punctuations
# Remove non-alphabetic characters

review_example = re.sub("[^a-zA-Z]",' ',review_example)
review_example


# In[44]:


review_example = review_example.split()


# In[45]:


# convert the texts to lowercase to avoid our ml percieving capital letters as a different work

review_example


# In[46]:


rt = lambda x: re.sub("[^a-zA-Z]", ' ',str(x))
df["reviewText"] = df["reviewText"].map(rt)
df["reviewText"] = df["reviewText"].str.lower()
df.head()


# The above code applies three operations to the reviewText column of a DataFrame df:
# 
# lambda x: re.sub("[^a-zA-Z]", ' ',str(x)): This lambda function replaces all non-letter characters in a string with a space character. The re.sub() function is used to replace any character that does not match the pattern [a-zA-Z] (i.e., any non-letter character) with a space. The str() function is used to convert the input x to a string in case it is not already a string. This lambda function is applied to each value in the reviewText column using the map() method.
# 
# df["reviewText"].str.lower(): This converts all the text in the reviewText column to lowercase using the str.lower() method.

# #### Sentiment Analysis
# 
# In sentiment analysis, we are trying to determine the mood of the comment

# In[49]:


#Import our library

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[57]:


df[['polarity', 'subjectivity']]= df['reviewText'].apply(lambda Text:pd.Series(TextBlob(Text).sentiment)) # textblob will return polarity and subjectivity
#polarity indicates the mode of the comment, whether positive or negative, the closer to the one the more positive and zero -negative
for index, row in df['reviewText'].iteritems():
    
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    if neg>pos:
        df.loc[index, 'sentiment']= 'Negative'
    elif pos > neg:
        df.loc[index, 'sentiment'] = 'Positive'
    else:
        df.loc[index, 'sentiment'] = 'Neutral'


# In[58]:


#identify the 20 interpretation 
df[df['sentiment']=='Positive'].sort_values('wilson_lower_bound',
                                           ascending =False).head(5)


# In[59]:


# Ploting thr unbalance data problem, to categorize it into positive, negative and neutral

categorical_variable_summary(df,'sentiment')

