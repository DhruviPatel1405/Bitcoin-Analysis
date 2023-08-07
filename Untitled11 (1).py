#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


# In[4]:


df=pd.read_csv('bitcoin_price.csv')


# In[5]:


df


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


df.columns


# In[9]:


df.info()


# In[10]:


df.describe().T


# In[11]:


df.duplicated().sum()


# In[12]:


df.isnull().sum()


# In[13]:


df.dtypes


# In[14]:


type(df['Date'][0])


# In[15]:


df['Date']= pd.to_datetime(df['Date'])


# In[16]:


df['Date'].dtypes


# In[17]:


df['Date'].min()


# In[18]:


df['Date'].max()


# In[19]:


df.sort_index(ascending=False).reset_index()


# In[20]:


data = df.sort_index(ascending=False).reset_index()


# In[21]:


data.drop('index',axis=1,inplace=True)


# In[22]:


data.columns


# In[23]:


plt.figure(figsize=(20,12))
for index,col in enumerate(['Open', 'High', 'Low', 'Close'],1):
    plt.subplot(2,2,index)
    plt.plot(df['Date'],df[col])
    plt.title(col)


# In[24]:


data.shape


# In[25]:


bitcoin_sample = data[0:50]


# In[26]:


get_ipython().system('pip install chart_studio')
get_ipython().system('pip install plotly')


# In[27]:


import chart_studio.plotly as py


# In[28]:


import plotly.graph_objs as go


# In[29]:


import plotly.express as px


# In[30]:


from plotly.offline import download_plotlyjs,init_notebook_mode , plot , iplot


# In[31]:


init_notebook_mode(connected=True)


# In[32]:


trace = go.Candlestick(x=bitcoin_sample['Date'],
              high =bitcoin_sample['High'] ,
               open =bitcoin_sample['Open'] ,
               close =bitcoin_sample['Close'],
               low =bitcoin_sample['Low'] 
              )
candle_data = [trace]
layout={
    'title': 'Bitcoin Historical Price',
'xaxis':{'title':'Date'}
}
fig = go.Figure(data = candle_data,layout=layout)
fig.update_layout(xaxis_rangeslider_visible = False)
fig.show()


# In[33]:


data['Close'].plot()


# In[35]:


data.set_index('Date',inplace = True)


# In[36]:


data['Close'].plot()


# In[39]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
data['Close'].plot()
plt.title('no scaling')

plt.subplot(1,2,2)
np.log1p(data['Close']).plot()
plt.title('Log scaling')


# In[40]:


data.head(4)


# In[41]:


data['Close'].resample('Y').mean()


# In[42]:


data['Close'].resample('Y').mean().plot()


# In[43]:


data['Close'].resample('Q').mean()


# In[44]:


data['Close'].resample('Q').mean().plot()


# In[45]:


data['Close'].resample('M').mean()


# In[46]:


data['Close'].resample('M').mean().plot()


# In[47]:


data['Close_price_pct_change'] = data['Close'].pct_change()*100


# In[48]:


data['Close_price_pct_change']


# In[49]:


data['Close_price_pct_change'].plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




