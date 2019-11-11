
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


os.chdir('D:\\Analytics vidya ML course\\Module_14_Assignment_-_Dataset_Solution\\Module 14 Assignment - Dataset+Solution')


# In[3]:


df = pd.read_csv('Train_pjb2QcD.csv')


# In[4]:


df.head()


# In[5]:


df = df.drop(['ID', 'Office_PIN', 'Applicant_City_PIN'], axis  =1)


# In[6]:


a = df['Applicant_Gender'].value_counts().sort_values(ascending = False)


# In[7]:


sns.barplot(a.index, a.values)


# In[8]:


df.head()


# In[9]:


pd.crosstab(df['Business_Sourced'], df['Manager_Current_Designation'])


# In[10]:


plt.figure(figsize = (8, 12))
pd.crosstab(df['Business_Sourced'], df['Manager_Current_Designation']).plot(kind = 'bar')


# In[11]:


pd.crosstab(df['Business_Sourced'], df['Manager_Grade'])


# In[12]:


plt.figure(figsize = (8, 12))
pd.crosstab(df['Business_Sourced'], df['Manager_Grade']).plot(kind = 'bar')


# In[13]:


df['Business_Sourced'].value_counts()


# In[14]:


df.pivot_table(values = 'Business_Sourced', index = 'Applicant_Gender', columns = 'Applicant_Marital_Status')


# In[15]:


pd.crosstab(df['Applicant_Gender'], df['Applicant_Qualification'])


# In[16]:


df.isnull().sum()


# In[17]:


df['Applicant_Gender'].value_counts()


# In[18]:


df['Applicant_Gender'] = df['Applicant_Gender'].fillna('M')


# In[19]:


df['Applicant_Gender'].isnull().sum()


# In[20]:


df['Applicant_Marital_Status'] = df['Applicant_Marital_Status'].fillna('M')


# In[21]:


df['Applicant_Occupation'].value_counts()


# In[22]:


df['Applicant_Marital_Status'].value_counts()


# In[23]:


df.isnull().sum()


# In[24]:


df['Applicant_Occupation'] = df['Applicant_Occupation'].fillna('Salaried')


# In[25]:


df['Applicant_Qualification'].value_counts()


# In[26]:


df['Applicant_Qualification'] = df['Applicant_Qualification'].fillna('Class XII')


# In[27]:


df.isnull().sum()


# In[28]:


from sklearn.model_selection import train_test_split
x = df.drop(['Business_Sourced'], axis = 1)
y = df['Business_Sourced']


# In[29]:


corr = df.corr()
corr['Business_Sourced'].sort_values(ascending = False)


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 101, stratify = y)


# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[32]:


df.describe(include = 'all')


# In[33]:


#df['Applicant_Gender'] = pd.get_dummies(df['Applicant_Gender'])


# In[34]:


df.head()


# In[35]:


from sklearn.preprocessing import LabelEncoder


# In[36]:


le = LabelEncoder()
le.fit_transform(['M', 'F'])


# In[37]:


df.head()


# In[38]:


df['Applicant_Gender'] = df['Applicant_Gender'].map({
    'M':1,
    'F':0
})


# In[39]:


df.head()


# In[40]:


df['Applicant_Marital_Status'].unique()


# In[41]:


df['Applicant_Marital_Status'] = df['Applicant_Marital_Status'].map({
    'M':1,
    'S':2,
    'W':3,
    'D':4
})


# In[42]:


df['Applicant_Occupation'].unique()


# In[43]:


df['Applicant_Occupation']= df['Applicant_Occupation'].map({
    'Business':1,
    'Self Employed':2,
    'Salaried':3,
    'Others':4,
    'Student':5
})


# In[44]:


df['Applicant_Qualification'].unique()


# In[45]:


le.fit_transform(df['Applicant_Qualification'].unique())


# In[46]:


df['Applicant_Qualification'] = le.fit_transform(df['Applicant_Qualification'])


# In[47]:


df['Applicant_Qualification'].head()


# In[48]:


df.head()


# In[49]:


df['Manager_Joining_Designation'].unique()


# In[50]:


le.fit_transform(df['Manager_Joining_Designation'].unique())


# In[51]:


df['Manager_Joining_Designation'] = df['Manager_Joining_Designation'].map({
    'Level 1':0,
    'Level 2':1,
    'Level 3':2,
    'Level 4':3, 
    'Level 5':4, 
    'Level 6':5,
    'Level 7':6,
    'Other':7
})


# In[52]:


df['Manager_Joining_Designation'].head()


# In[53]:


df.head()


# In[54]:


df['Manager_Current_Designation'].unique()


# In[55]:


df['Manager_Current_Designation'] = df['Manager_Current_Designation'].map({
    'Level 1':0,
    'Level 2':1,
    'Level 3':2,
    'Level 4':3,
    'Level 5':4,
})


# In[56]:


df.head()


# In[57]:


df['Manager_Status'].unique()


# In[58]:


df['Manager_Status'] = le.fit_transform(df['Manager_Status'])


# In[59]:


df['Manager_Gender'] = df['Manager_Gender'].map({
    'M':1, 
    'F':0
})


# In[60]:


x = df.drop(['Business_Sourced'],axis= 1)


# In[61]:


y = df['Business_Sourced']


# In[62]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()


# In[63]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_Train, y_test = train_test_split(x, y, random_state = 101, stratify = y)


# In[64]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[65]:


model1 = LogisticRegression()
model1.fit(x_train, y_train)
pred1 = model1.predict(x_test)
model1.score(x_train, y_train)


# In[66]:


pred1_ = model1.predict(x_train)
pred1_


# In[67]:


from sklearn.metrics import f1_score
f1_score(pred1, y_test)


# In[68]:


f1_score(y_train, pred1_)


# In[69]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pred1_, y_train)


# In[70]:


cm


# In[71]:


model1.coef_


# In[72]:


plt.figure(figsize = (8, 12))
c = model1.coef_.reshape(-1)
d = range(len(x_train.columns))
plt.bar(d, c)
plt.xlabel('Coefficients')
plt.ylabel('Variables')


# In[73]:


coeff = pd.DataFrame({
    'variables':x_train.columns,
    'coeff':abs(c)
    
})


# In[74]:


coeff.head()


# In[75]:


s = coeff[coeff['coeff']>-0.3]


# In[76]:


subset = df[s['variables'].values]


# In[77]:


subset.head()


# In[78]:


m = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(subset, y, random_state = 101, stratify = y)


# In[79]:


s


# In[80]:


m.fit(x_train, y_train)


# In[81]:


m.score(x_train, y_train)


# In[82]:


pred1 = m.predict(x_test)


# In[83]:


from sklearn.metrics import f1_score
f1_score(pred1, y_test)


# In[84]:


plt.figure(figsize = (8, 12))
c = m.coef_.reshape(-1)
d = range(len(x_train.columns))
plt.bar(d, c)
plt.xlabel('Coefficients')
plt.ylabel('Variables')


# # Decision Tree

# In[85]:


from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier(random_state=10)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 101, stratify = y)


# In[86]:


model2.fit(x_train, y_train)
pred2 = model2.predict(x_test)
pred2_ = model2.predict(x_train)
model2.score(x_train, y_train)


# In[87]:


from sklearn.metrics import accuracy_score
test_score = accuracy_score(pred2, y_test)
train_score = accuracy_score(pred2_, y_train)


# In[88]:


test_score, train_score


# In[89]:


cm = confusion_matrix(pred2, y_test)
cm1 = confusion_matrix(pred2_, y_train)
cm


# In[90]:


cm1


# In[91]:


train_accuracy = []
test_accuracy = []

for depth in range(1, 20):
    model2 = DecisionTreeClassifier(max_depth = depth, random_state = 10)
    model2.fit(x_train, y_train)
    train_accuracy.append(model2.score(x_train, y_train))
    test_accuracy.append(model2.score(x_test, y_test))


# In[92]:


dataframe = pd.DataFrame({
    'depth':range(1,20),
    'train_accuracy':train_accuracy,
    'test_accuracy':test_accuracy
})


# In[93]:


plt.plot(dataframe['depth'], dataframe['train_accuracy'], marker = 'o')
plt.plot(dataframe['depth'], dataframe['test_accuracy'], marker = 'o')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.legend()


# In[94]:


model2 = DecisionTreeClassifier(max_depth = 6, max_leaf_nodes=20, random_state = 10)
model2.fit(x_train, y_train)
model2.score(x_train, y_train)
pred2 = model2.predict(x_test)


# In[95]:


from sklearn.metrics import f1_score
f1_score(pred2, y_test)


# # K-NN model

# In[96]:


from sklearn.neighbors import KNeighborsClassifier
model3 = KNeighborsClassifier(n_neighbors=5)
model3.fit(x_train, y_train)
model3.score(x_train, y_train)


# In[97]:


model3.score(x_test, y_test)


# In[98]:


from sklearn.metrics import f1_score
def elbow(k):
    test_error = []
    for i in k:
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(x_train, y_train)
        temp = clf.predict(x_test)
        temp = f1_score(temp, y_test)
        error = 1-temp
        test_error.append(error)
    return test_error


# In[99]:


k = range(6, 20, 2)


# In[100]:


test = elbow(k)


# In[101]:


#plt.figure(figsize = (8, 12))
plt.plot(k, test)
plt.xlabel('K')
plt.ylabel('Error')
plt.plot()


# In[102]:


model3 = KNeighborsClassifier(n_neighbors=8)
model3.fit(x_train, y_train)
model3.score(x_train, y_train)
pred3 = model3.predict(x_test)


# In[103]:


from sklearn.metrics import f1_score
f1_score(pred3, y_test)


# In[104]:


from statistics import mode
final_pred = np.array([])
for i in range(0, len(x_test)):
    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))


# In[105]:


final_pred


# In[106]:


f = pd.DataFrame(final_pred)


# In[107]:


f[0].value_counts()

