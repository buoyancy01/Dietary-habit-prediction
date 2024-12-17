#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing the dataset
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
data = pd.read_csv("C:\\Users\\AJIBOLA\\Downloads\\Health.csv")
data.dropna()

X = data[["Calories Burned", 'Sleep Quality']]
y = data['Dietary Habits']

#Encode the dataset
X = pd.get_dummies(X, drop_first = True)
le = LabelEncoder()
y = le.fit_transform(y)

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100, test_size = 0.2)

#Building the model
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#Using a dictionary
#word_dict = {2:'healthy', 0:'Unhealthy'}
#X_test = 

#Model prediction
prediction = model.predict(X_test)

decoder = le.inverse_transform(prediction)

#check accuracy
from sklearn.metrics import accuracy_score
accurate = accuracy_score(y_test, prediction)


#Save the model
#import joblib
#saved_model = joblib.dump(model,'Check your Dietary habit.joblib')
#saved_model



# In[15]:





# In[ ]:




