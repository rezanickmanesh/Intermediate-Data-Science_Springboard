
# coding: utf-8

# # JSON examples and exercise
# ****
# + get familiar with packages for dealing with JSON
# + study examples with JSON strings and files 
# + work on exercise to be completed and submitted 
# ****
# + reference: http://pandas.pydata.org/pandas-docs/stable/io.html#io-json-reader
# + data source: http://jsonstudio.com/resources/
# ****

# In[68]:


import pandas as pd


# ## imports for Python, Pandas

# In[69]:


import json
from pandas.io.json import json_normalize


# ## JSON example, with string
# 
# + demonstrates creation of normalized dataframes (tables) from nested json string
# + source: http://pandas.pydata.org/pandas-docs/stable/io.html#normalization

# In[70]:


# define json string
data = [{'state': 'Florida', 
         'shortname': 'FL',
         'info': {'governor': 'Rick Scott'},
         'counties': [{'name': 'Dade', 'population': 12345},
                      {'name': 'Broward', 'population': 40000},
                      {'name': 'Palm Beach', 'population': 60000}]},
        {'state': 'Ohio',
         'shortname': 'OH',
         'info': {'governor': 'John Kasich'},
         'counties': [{'name': 'Summit', 'population': 1234},
                      {'name': 'Cuyahoga', 'population': 1337}]}]


# In[71]:


# use normalization to create tables from nested element
json_normalize(data, 'counties')


# In[72]:


# further populate tables created from nested element
json_normalize(data, 'counties', ['state', 'shortname', ['info', 'governor']])


# ****
# ## JSON example, with file
# 
# + demonstrates reading in a json file as a string and as a table
# + uses small sample file containing data about projects funded by the World Bank 
# + data source: http://jsonstudio.com/resources/

# In[73]:


# load json as string
json.load((open('C:\Users\Rezanick\Desktop\Projects\Springboard\Exercise 2\data_wrangling_json\data\world_bank_projects_less.json')))


# In[74]:


# load as Pandas dataframe
sample_json_df = pd.read_json('C:\Users\Rezanick\Desktop\Projects\Springboard\Exercise 2\data_wrangling_json\data\world_bank_projects_less.json')
sample_json_df


# ****
# ## JSON exercise
# 
# Using data in file 'data/world_bank_projects.json' and the techniques demonstrated above,
# 1. Find the 10 countries with most projects
# 2. Find the top 10 major project themes (using column 'mjtheme_namecode')
# 3. In 2. above you will notice that some entries have only the code and the name is missing. Create a dataframe with the missing names filled in.

# In[75]:


import os
import numpy as np
import json
os.chdir('C:\Users\Rezanick\Desktop\Projects\Springboard\Exercise 2\data_wrangling_json\data')

with open('world_bank_projects.json', 'r') as json_file:
    json_file=json.load(json_file)    


# In[76]:


type(json_file)


# In[77]:


import pandas as pd
import json
from pandas.io.json import json_normalize


# In[78]:


df=pd.read_json('C:\Users\Rezanick\Desktop\Projects\Springboard\Exercise 2\data_wrangling_json\data\world_bank_projects.json')
df.head()


# In[79]:


wbp_df["countryname"].value_counts()[0:10] 


# In[80]:


D=wbp_df["countryshortname"].unique()
D.sort()
D


# In[81]:


df['project_name']


# In[82]:


df['project_name'].groupby(df['countryshortname']).count().sort_values(ascending=False).head(10)


# In[83]:


data = json.load((open('C:\Users\Rezanick\Desktop\Projects\Springboard\Exercise 2\data_wrangling_json\data\world_bank_projects.json')))
pt = json_normalize(data, 'mjtheme_namecode')
print(pt)
pt.name.value_counts().head(10)


# In[84]:


pt = pt.sort_values(by=['code','name'])
pt


# In[85]:


pt['name'].replace('', np.nan, inplace=True)
pt['name'] = pt['name'].bfill()
pt

