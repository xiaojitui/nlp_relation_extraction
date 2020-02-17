#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openie import StanfordOpenIE


# In[2]:


with StanfordOpenIE() as client:
    text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
    print('Text: %s.' % text)
    for triple in client.annotate(text):
        print('|-', triple)


# In[3]:


triple


# In[4]:


client.annotate(text)


# In[ ]:





# In[ ]:





# In[11]:


# examples: 
text1 = "Sep 3, 2013 - Microsoft agrees to buy Nokia's mobile phone business for $7.2 billion."
text2 = 'Instant view: Private sector adds 114,000 jobs in July: ADP.'
text3 = 'Private sector adds 114,000 jobs in July'


# In[6]:


client.annotate(text1)


# In[10]:


client.annotate(text2)


# In[12]:


client.annotate(text3)


# In[ ]:





# In[14]:


import nltk
from nltk.stem import PorterStemmer


# In[15]:


porter = PorterStemmer()


# In[18]:


porter.stem('adds')
porter.stem('Private sector')
porter.stem('114,000 jobs')


# In[ ]:





# In[ ]:





# In[19]:


for ele in '114,000 jobs in July'.split():
    print(ele, porter.stem(ele))


# In[ ]:





# In[26]:


nltk.download()


# In[29]:


nltk.download()


# In[ ]:





# In[ ]:





# In[27]:


from nltk import word_tokenize
t = word_tokenize('114,000 jobs in July')
t


# In[ ]:





# In[30]:


nltk.pos_tag(t)


# In[ ]:





# In[ ]:





# In[33]:


nltk.download()


# In[34]:


from nltk.corpus import verbnet


# In[38]:


verbnet.classids(lemma = 'add')


# In[39]:


verbnet.classids(lemma = 'buy')


# In[41]:


verbnet.classids(lemma = 'take')


# In[42]:


verbnet.classids(lemma = 'give')


# In[44]:


verbnet.classids(lemma = 'like')


# In[ ]:





# In[ ]:




