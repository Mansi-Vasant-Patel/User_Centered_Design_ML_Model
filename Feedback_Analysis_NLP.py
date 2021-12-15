#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt, re
import io
import nltk


# In[2]:


df = pd.read_csv(r'C:\Users\utkar\OneDrive\Desktop\Machine Learning\Customerfeedback\Restaurant_Reviews.tsv', delimiter = '\t')


# In[3]:


df.head(1005)


# In[4]:


feedback = df


# In[5]:


feedback.head()


# In[6]:


import nlp_tools
import contractions


# In[7]:


feedback['cleanReviews'] = feedback['Review'].apply(contractions.expand_contraction)


# In[8]:


feedback['cleanReviews'] = feedback['cleanReviews'].apply(nlp_tools.lemmatization_sentence)


# In[9]:


like_dislike_list = feedback['cleanReviews'].tolist()


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer


# In[11]:


cv = CountVectorizer()


# In[138]:


cv = cv.fit(like_dislike_list)


# In[139]:


filename = r'C:\Users\utkar\OneDrive\Desktop\Machine Learning\Customerfeedback\count_vectorizer.pkl'


# In[140]:


pickle.dump(cv, open(filename, 'wb'))


# In[13]:


X = cv.transform(like_dislike_list).toarray()


# In[14]:


len(cv.get_feature_names())


# In[15]:


y = feedback['Liked'].values


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y)


# In[17]:


import lazypredict
from lazypredict.Supervised import LazyClassifier


# In[18]:


clf = LazyClassifier(verbose=0,ignore_warnings=True)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
models


# In[19]:


from sklearn.linear_model import LogisticRegression


# In[20]:


reg = LogisticRegression(random_state = 0)


# In[21]:


model= reg.fit(x_train,y_train)


# In[22]:


predicted_values = model.predict(x_test)


# In[23]:


predicted_values


# In[24]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, predicted_values)


# In[25]:


bad_review = 'The meal was not fullfiling'

clean_review = contractions.expand_contraction(bad_review)
lemma_review = nlp_tools.lemmatization_sentence(clean_review)
vector_review = cv.transform([lemma_review]).toarray()
review = model.predict(vector_review)
pred = model.predict_proba(vector_review)


# In[26]:


pred


# In[27]:


review


# In[28]:


good_review = 'The meal was satisfying'

clean_review = contractions.expand_contraction(good_review)
lemma_review = nlp_tools.lemmatization_sentence(clean_review)
vector_review = cv.transform([lemma_review]).toarray()
review = model.predict(vector_review)
pred = model.predict_proba(vector_review)


# In[29]:


pred


# In[30]:


review


# In[33]:


import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Speak Anything :")
    audio = r.listen(source)
    
    try:
        text = r.recognize_google(audio)
        
        print("You said : {}".format(text))
    except:
        print("Sorry could not recognize what you said")
        


# In[34]:


clean_review = contractions.expand_contraction(text)
lemma_review = nlp_tools.lemmatization_sentence(clean_review)
vector_review = cv.transform([lemma_review]).toarray()
review = model.predict(vector_review)
pred = model.predict_proba(vector_review)


# In[35]:


pred


# In[36]:


review


# In[39]:


df.head(1005)


# In[40]:


df.loc[len(df), 'Review'] = text
df.loc[len(df) - 1, 'Liked'] = review[0]


# In[41]:


df.head(1005)


# In[42]:


feedback = df


# In[46]:


df = df.drop('cleanReviews', 1)


# In[117]:


fields = [text, review[0]]


# In[118]:


import csv


# In[120]:


with open(r'C:\Users\utkar\OneDrive\Desktop\Machine Learning\Customerfeedback\Restaurant_Reviews.tsv','a') as fd:
    writer = csv.writer(fd, delimiter='\t')
    writer.writerow(fields)


# In[121]:


feedback.tail()


# In[122]:


df.tail()


# In[123]:


import pickle


# In[124]:


filename = r'C:\Users\utkar\OneDrive\Desktop\Machine Learning\Customerfeedback\feedbackmodel.pkl'


# In[125]:


pickle.dump(model, open(filename, 'wb'))


# In[126]:


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)


# In[127]:


prediction = loaded_model.predict(x_test)


# In[128]:


prediction


# In[130]:


import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Speak Anything :")
    audio = r.listen(source)
    
    try:
        text = r.recognize_google(audio)
        
        print("You said : {}".format(text))
    except:
        print("Sorry could not recognize what you said")
        


# In[131]:


clean_review = contractions.expand_contraction(text)
lemma_review = nlp_tools.lemmatization_sentence(clean_review)
vector_review = cv.transform([lemma_review]).toarray()
review = loaded_model.predict(vector_review)
pred = loaded_model.predict_proba(vector_review)


# In[132]:


review


# In[133]:


pred


# In[134]:


df.loc[len(df), 'Review'] = text
df.loc[len(df) - 1, 'Liked'] = review[0]


# In[135]:


df.tail()


# In[ ]:





# In[ ]:




