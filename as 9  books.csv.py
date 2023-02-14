#!/usr/bin/env python
# coding: utf-8

# In[22]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[23]:


# Import Dataset
book=pd.read_csv('book.csv')
book


# In[24]:


# Data preprocessing not required as it is already in transaction format


# ## Apriori Algorithm

# ### 1. Association rules with 10% Support and 70% confidence

# In[25]:


# With 10% Support
frequent_itemsets=apriori(book,min_support=0.1,use_colnames=True)
frequent_itemsets


# In[26]:


# with 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules


# In[27]:


## A leverage value of 0 indicates independence. Range will be [-1 1]
## A high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]


# In[28]:


rules.sort_values('lift',ascending=False)


# In[29]:


# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]


# In[30]:


# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# ### 2. Association rules with 20% Support and 60% confidence

# In[31]:


# With 20% Support
frequent_itemsets2=apriori(book,min_support=0.20,use_colnames=True)
frequent_itemsets2


# In[32]:


# With 60% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
rules2


# In[40]:


# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# ### 3. Association rules with 5% Support and 80% confidence

# In[34]:


# With 5% Support
frequent_itemsets3=apriori(book,min_support=0.05,use_colnames=True)
frequent_itemsets3


# In[35]:


# With 80% confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.8)
rules3


# In[36]:


rules3[rules3.lift>1]


# In[37]:


# visualization of obtained rule
plt.scatter(rules3['support'],rules3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# In[ ]:




