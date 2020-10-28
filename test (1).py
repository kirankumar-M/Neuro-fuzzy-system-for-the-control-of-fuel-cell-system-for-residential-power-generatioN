#!/usr/bin/env python
# coding: utf-8

# In[110]:


#Please use ANSIF module for this program to run
import numpy as np
import pandas as pd
import random

a=np.random.uniform(low=0,high=14,size=(50,1))
df=pd.DataFrame(a,columns=['production'])
df['demand']=np.random.uniform(low=16,high=(16+np.random.randint(1,3)),size=(50,1))

df['power_to_be_generated']=(df['demand']-df['production'])
print(df)
#Here all values represent Power in units of MW
#Demand is the the power required by the houses
#Production is the power produced by the fuel cells 
#Power_to_be_generated is the more power that needed to be produced by the fuel cells to meet the demand requirements


# In[111]:


mf=[[['gaussmf',{"mean":np.mean(np.arange(0,8)),
                "sigma":np.std(np.arange(0,8))}],
    ['gaussmf',{"mean":np.mean(np.arange(7,15)),
                "sigma":np.std(np.arange(7,15))}],
     ['gaussmf',{"mean":np.mean(np.arange(14,19)),
                "sigma":np.std(np.arange(14,19))}]],
   [['gaussmf',{"mean":np.mean(np.arange(0,8)),
                "sigma":np.std(np.arange(0,8))}],
    ['gaussmf',{"mean":np.mean(np.arange(7,15)),
                "sigma":np.std(np.arange(7,15))}],
     ['gaussmf',{"mean":np.mean(np.arange(14,19)),
                "sigma":np.std(np.arange(14,19))}]]]


# In[112]:


from membership import membershipfunction
mfc=membershipfunction.MemFuncs(mf)


# In[113]:


power_to_be_generated=df.pop('power_to_be_generated')


# In[114]:


import anfis
anf=anfis.ANFIS(df,power_to_be_generated,mfc)
train=anf.trainHybridJangOffLine(epochs=20)


# In[115]:


anf.plotErrors()


# In[116]:


anf.plotResults()


# In[104]:


print(train)


# In[ ]:




