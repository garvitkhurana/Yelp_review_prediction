
# coding: utf-8

# # Tutorial Exercise: Yelp reviews

# ## Introduction
# 
# This exercise uses a small subset of the data from Kaggle's [Yelp Business Rating Prediction](https://www.kaggle.com/c/yelp-recsys-2013) competition.
# 
# **Description of the data:**
# 
# - **`yelp.csv`** contains the dataset. It is stored in the repository (in the **`data`** directory), so there is no need to download anything from the Kaggle website.
# - Each observation (row) in this dataset is a review of a particular business by a particular user.
# - The **stars** column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.
# - The **text** column is the text of the review.
# 
# **Goal:** Predict the star rating of a review using **only** the review text.
# 
# **Tip:** After each task, I recommend that you check the shape and the contents of your objects, to confirm that they match your expectations.

# ## Task 1
# 
# Read **`yelp.csv`** into a pandas DataFrame and examine it.

# In[1]:


import pandas as pd
data=pd.read_csv("/Users/garvitkhurana/Desktop/yelp.csv")


# ## Task 2
# 
# Create a new DataFrame that only contains the **5-star** and **1-star** reviews.
# 
# - **Hint:** [How do I apply multiple filter criteria to a pandas DataFrame?](http://nbviewer.jupyter.org/github/justmarkham/pandas-videos/blob/master/pandas.ipynb#9.-How-do-I-apply-multiple-filter-criteria-to-a-pandas-DataFrame%3F-%28video%29) explains how to do this.

# In[2]:


new_data=data[(data.stars==5) | (data.stars==1)]


# ## Task 3
# 
# Define X and y from the new DataFrame, and then split X and y into training and testing sets, using the **review text** as the only feature and the **star rating** as the response.
# 
# - **Hint:** Keep in mind that X should be a pandas Series (not a DataFrame), since we will pass it to CountVectorizer in the task that follows.

# In[3]:


X=new_data.text
y=new_data.stars
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
org_X=X_test


# ## Task 4
# 
# Use CountVectorizer to create **document-term matrices** from X_train and X_test.

# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X_train=vect.fit_transform(X_train)
X_test=vect.transform(X_test)


# ## Task 5
# 
# Use multinomial Naive Bayes to **predict the star rating** for the reviews in the testing set, and then **calculate the accuracy** and **print the confusion matrix**.
# 
# - **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains how to interpret both classification accuracy and the confusion matrix.

# In[5]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
pred=clf.predict(X_test)


# In[6]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(pred,y_test)


# In[7]:


confusion_matrix(pred,y_test)


# ## Task 6 (Challenge)
# 
# Calculate the **null accuracy**, which is the classification accuracy that could be achieved by always predicting the most frequent class.
# 
# - **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains null accuracy and demonstrates two ways to calculate it, though only one of those ways will work in this case. Alternatively, you can come up with your own method to calculate null accuracy!

# In[8]:


y_test.value_counts().head(1)/len(y_test)


# ## Task 7 (Challenge)
# 
# Browse through the review text of some of the **false positives** and **false negatives**. Based on your knowledge of how Naive Bayes works, do you have any ideas about why the model is incorrectly classifying these reviews?
# 
# - **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains the definitions of "false positives" and "false negatives".
# - **Hint:** Think about what a false positive means in this context, and what a false negative means in this context. What has scikit-learn defined as the "positive class"?

# In[9]:


false_positive=org_X[y_test<pred]###org_X= Orignal X_test data before CountVectorizer


# In[10]:


false_negative=org_X[y_test>pred]


# ## Task 8 (Challenge)
# 
# Calculate which 10 tokens are the most predictive of **5-star reviews**, and which 10 tokens are the most predictive of **1-star reviews**.
# 
# - **Hint:** Naive Bayes automatically counts the number of times each token appears in each class, as well as the number of observations in each class. You can access these counts via the `feature_count_` and `class_count_` attributes of the Naive Bayes model object.

# In[11]:


tokens=vect.get_feature_names()


# In[12]:


one_star=clf.feature_count_[0][:]


# In[13]:


five_star=clf.feature_count_[1][:]


# In[14]:


tokens_frame = pd.DataFrame({'token':tokens, 'one_star':one_star, 'five_star':five_star}).set_index('token')
tokens_frame['one_star'] = tokens_frame.one_star + 1
tokens_frame['five_star'] = tokens_frame.five_star + 1
##
tokens_frame['one_star'] = tokens_frame.one_star / clf.class_count_[0]
tokens_frame['five_star'] = tokens_frame.five_star / clf.class_count_[1]
##
tokens_frame['five_star_ratio'] = tokens_frame.five_star / tokens_frame.one_star
tokens_frame.sort_values('five_star_ratio', ascending=False).head(10)
##


# In[15]:


tokens_frame.sort_values('five_star_ratio', ascending=True).head(10)


# ## Task 9 (Challenge)
# 
# Up to this point, we have framed this as a **binary classification problem** by only considering the 5-star and 1-star reviews. Now, let's repeat the model building process using all reviews, which makes this a **5-class classification problem**.
# 
# Here are the steps:
# 
# - Define X and y using the original DataFrame. (y should contain 5 different classes.)
# - Split X and y into training and testing sets.
# - Create document-term matrices using CountVectorizer.
# - Calculate the testing accuracy of a Multinomial Naive Bayes model.
# - Compare the testing accuracy with the null accuracy, and comment on the results.
# - Print the confusion matrix, and comment on the results. (This [Stack Overflow answer](http://stackoverflow.com/a/30748053/1636598) explains how to read a multi-class confusion matrix.)
# - Print the [classification report](http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report), and comment on the results. If you are unfamiliar with the terminology it uses, research the terms, and then try to figure out how to calculate these metrics manually from the confusion matrix!

# In[16]:


import pandas as pd
data=pd.read_csv("/Users/garvitkhurana/Desktop/yelp.csv")


# In[17]:


X = data.text
y = data.stars


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[19]:


X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)


# In[20]:


from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier(n_estimators=206)
clf.fit(X_train_dtm, y_train)


# In[21]:


pred=clf.predict(X_test_dtm)


# In[22]:


accuracy_score(y_test, pred)


# In[23]:


y_test.value_counts().head(1) / y_test.shape##null accuracy


# In[24]:


confusion_matrix(y_test, pred)


# In[25]:


print(classification_report(y_test, pred))

