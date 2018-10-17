
# coding: utf-8

# # Enron Machine Learing Project - Tianxing Zhai

# ## 1. Exploratory Data Analysis

# ### 1.1 Load and transform data

# In[1]:


import pandas as pd
import numpy as np
import pickle


# In[14]:


enron_dict = pickle.load(open("final_project_dataset.pkl", "rb"))


# In[15]:


len(enron_dict)


# Let's print the first two elments in the enron dictionary:

# In[16]:


i = 0
for k,v in enron_dict.items():
    print(k)
    print(v)
    i += 1
    if i > 1:
        break


# The dictionary is a nested dictionary. In the first level, the key is staff names of Enron and the value is another dictionary which records all attributes of this person.

# There are some outliers here. The first one named 'TOTAL'.

# In[17]:


enron_dict['TOTAL']


# In[18]:


del enron_dict['TOTAL']


# The second one named 'THE TRAVEL AGENCY IN THE PARK'.

# In[19]:


enron_dict['THE TRAVEL AGENCY IN THE PARK']


# In[20]:


del enron_dict['THE TRAVEL AGENCY IN THE PARK']


# The third one named 'LOCKHART EUGENE E'(all features are missing).

# In[21]:


enron_dict['LOCKHART EUGENE E']


# In[22]:


del enron_dict['LOCKHART EUGENE E']


# In[23]:


len(enron_dict)


# Then transform the dictionary to pandas dataframe which is easier to analyze 

# In[24]:


name_list = []
attr_list = []

for name,attr in enron_dict.items():
    name_list.append(name)
    attr_list.append(pd.DataFrame.from_dict(attr, orient='index'))

enron = pd.concat(attr_list, keys = name_list).unstack()[0]


# In[25]:


enron.head()


# In[26]:


len(enron)


# In[27]:


enron.info()


# There are 143 persons and 21 columns.

# The data type of each column is not correct. All the columns, except 'email_address'(text) and 'poi'(bool), should be numeric type. I corrected the data type.

# In[28]:


emails = enron['email_address']
emails.replace('NaN', np.nan, inplace = True)
enron = enron.apply(pd.to_numeric, errors='coerce')
enron['email_address'] = emails


# In[29]:


enron.info()


# In[30]:


enron['poi'].sum()


# There are 18 persons are suspects, 125 persons are not suspects.

# As we can see, there are a lot of missing values. I will disscuss how to deal with by dividing the dataset into three classes: payments, stocks and emails. 

# In[31]:


payments_cols = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 
                 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments']
stocks_cols = ['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 
               'total_stock_value']
emails_clos = ['email_address', 'from_messages', 'from_poi_to_this_person', 'to_messages',
              'from_this_person_to_poi', 'shared_receipt_with_poi']

payments = enron[payments_cols]
stocks = enron[stocks_cols]
emails = enron[emails_clos]


# ### 1.2 Data Wrangling

# #### 1.2.1 Payments

# In[32]:


payments.head()


# In[33]:


payments.isna().sum()


# There are a lot of missing values in the payments dataset. But missing values here have very different meaning with missing values in other datasets. Each feature here, except 'total_payments', represents one kind of payment source. So if one feature is missing, it means that 'the source of payment is 0' rather than 'the source of payment is unknown'. Therefore, it is better to impute the missing value with 0.

# Another problem is that compositions of payments and the total payments may provide repetitive information. For example, people with higher total payments tends to have higher values in all compositions of payments. So it is better to transform absolute value of each composition to proportion of each composition.
# 

# In[34]:


payments_nums = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 
                 'loan_advances', 'other', 'expenses', 'director_fees']

payments_rate = payments.copy()
payments_rate = payments_rate.fillna(0)

for col in payments_nums:
    payments_rate[col] = payments_rate[col] / (payments_rate['total_payments'] - payments_rate['deferred_income'])
payments_rate = payments_rate.fillna(0)


# In[35]:


payments_rate.head()


# #### 1.2.2 Stocks

# In[36]:


stocks.head()


# First, I found some typos in the dataset:

# In[37]:


stocks.loc[(stocks['exercised_stock_options'] < 0) |
          (stocks['restricted_stock'] < 0) |
          (stocks['restricted_stock_deferred'] > 0)| 
          (stocks['total_stock_value'] < 0)]


# By looking back the pdf file with true values, I corrected them:

# In[38]:


ROBERT = pd.DataFrame({'exercised_stock_options': pd.Series([0], index = ['BELFER ROBERT']),
             'restricted_stock' : pd.Series([44093.0], index = ['BELFER ROBERT']),
             'restricted_stock_deferred' : pd.Series([-44093.0], index = ['BELFER ROBERT']),
             'total_stock_value' : pd.Series([0], index = ['BELFER ROBERT'])                                        
             })

SANJAY = pd.DataFrame({'exercised_stock_options': pd.Series([15456290.0], index = ['BHATNAGAR SANJAY']),
             'restricted_stock' : pd.Series([2604490.0], index = ['BHATNAGAR SANJAY']),
             'restricted_stock_deferred' : pd.Series([-2604490.0], index = ['BHATNAGAR SANJAY']),
             'total_stock_value' : pd.Series([15456290.0], index = ['BHATNAGAR SANJAY'])                                        
             })


# In[39]:


stocks.update(ROBERT)
stocks.update(SANJAY)


# In[40]:


stocks.loc[(stocks['exercised_stock_options'] < 0) |
          (stocks['restricted_stock'] < 0) |
          (stocks['restricted_stock_deferred'] > 0)| 
          (stocks['total_stock_value'] < 0)]


# Then I found that there many missing values as well. I imputed missing values and transformed the data using the same method as what I used in previous part.

# In[41]:


stocks.isna().sum()


# In[42]:


stocks_nums = ['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred']

stocks_rate = stocks.copy()
stocks_rate = stocks_rate.fillna(0)

for col in stocks_nums:
    stocks_rate[col] = stocks_rate[col] / (stocks_rate['total_stock_value'] - stocks_rate['restricted_stock_deferred'])
stocks_rate = stocks_rate.fillna(0)


# In[43]:


stocks_rate.head()


# #### 1.2.3 Emails

# In[44]:


emails.head()


# Emalis dataset has email address, counts on how many mails they sent and received, and how many of them are from and to poi. Finally, the count of shared receipt with poi is also included. 

# It is better to use the proportion of emails from and to pois rather than exact numbers. Besides, the email address gives us no useful information. It needs to be removed.

# In[45]:


emails_rate = pd.DataFrame()
emails_rate['from_poi_percent'] = emails['from_poi_to_this_person'] /  emails['to_messages']
emails_rate['to_poi_percent'] = emails['from_this_person_to_poi'] /  emails['from_messages']
emails_rate['shared_receipt_with_poi'] = emails['shared_receipt_with_poi']
emails_rate = emails_rate.fillna(0)


# In[46]:


emails_rate.head()


# #### 1.2.4 Combination

# The final step of data wrangling is to combine three cleaned datasets together as our final dataset. 

# In[47]:


enron_final = pd.concat([payments_rate, stocks_rate, emails_rate], axis = 1, sort = False)
enron_final['poi'] = enron['poi']


# In[48]:


enron_final.head()


# In[49]:


len(enron_final)


# In[50]:


enron_final.info()


# The final dataset contains 18 columns and 143 rows without missing values. I will use this to do futher analysis. 

# In[51]:


enron_final.to_pickle('enron_final.pkl')


# ## 2. Feature Selection and Optimization 

# ### 2.1 Feature Selection

# The final dataset contains 17 features, some of which are useless in predicting poi. So I need to select the most useful features. Here I will use univariate feature selection by computing the ANOVA p-values and selecting features whose p-values are less than 0.05.

# In[52]:


features = enron_final.drop(labels=["poi"], axis = 1)
labels = enron_final['poi']


# In[53]:


from sklearn.feature_selection import SelectKBest, f_classif


# In[54]:


selector = SelectKBest(f_classif)
selector.fit(features, labels)
features_new = selector.transform(features)


# In[55]:


pd.Series(list(features))[pd.Series(selector.pvalues_) < 0.05]


# There are 6 features have p values less than 0.05. So I will let k = 6.

# In[62]:


selector = SelectKBest(f_classif, k = 6)
selector.fit(features, labels)
features_new = selector.transform(features)


# In[63]:


features_new.shape


# In[64]:


pd.Series(list(features))[pd.Series(selector.get_support())]


# I selected these 6 features: 'bonus', 'loan_advances', 'total_payments', 'total_stock_value', 'to_poi_percent' and 'shared_receipt_with_poi'.

# To measure the effect of self-created features, I will create a new feature set with 'from_poi_percent', though it is not significantly associated with poi.

# In[65]:


selector = SelectKBest(f_classif, k = 7)
selector.fit(features, labels)
features_new_with_from = selector.transform(features)


# In[66]:


pd.Series(list(features))[pd.Series(selector.get_support())]


# ### 2.2 Feature Scaling

# Features here have different scale. Some algorithms like SVM do need feature scaling. I will use MinMaxScaler to scale features to [0,1].

# In[68]:


from sklearn.preprocessing import MinMaxScaler


# In[70]:


scaler = MinMaxScaler()
scaler.fit(features_new)
features_new_scaled = scaler.transform(features_new)


# In[71]:


scaler = MinMaxScaler()
scaler.fit(features_new_with_from)
features_new_with_from_scaled = scaler.transform(features_new_with_from)


# ## 3. Model Selection and Evaluation

# Here I will use these three machine learning algorithms: Naive Bayes, SVM, and Decision Tree. I will use GridSearchCV to tune parameters and select best model. I will use 'recall' as the scoring method. Because the dataset has much more negative values (not poi) than positive values (poi). Therefore, the ability for classifiers to find all the positive values is the most important. I will use StratifiedShuffleSplit as the cross-validation generator, becuase: The poi is binary variable, we need to make sure all groups after split have both poi and non-poi.

# In[84]:


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit


# In[85]:


sss = StratifiedShuffleSplit(n_splits=5, random_state=42)


# In[86]:


clf = GaussianNB()
clf.fit(features_new_scaled, labels)

best_gnb = clf
print(cross_val_score(best_gnb, features_new_scaled, y = labels, cv=sss, scoring = 'recall').mean())


# The Naive Bayes method does not have parameters to be tuned. So I use cross_val_score to evaluate the performance and print the mean recall score of cross validation.

# In[87]:


parameters = {'kernel':('linear', 'rbf'),
              'C': [1,10,100,1000,10000],
             'gamma': [0.1, 1, 10, 100]}

svc = SVC()
clf = GridSearchCV(svc, parameters, cv=sss, scoring='recall', n_jobs=-1)
clf.fit(features_new_scaled, labels)

best_svc = clf.best_estimator_
print(clf.best_score_ )


# In[89]:


parameters = {'min_samples_split': [2, 3, 4, 5]}

tree = DecisionTreeClassifier()
clf = GridSearchCV(tree, parameters, cv=sss, scoring='recall', n_jobs=-1)
clf.fit(features_new_scaled, labels)

best_tree = clf.best_estimator_
print(clf.best_score_ )


# The best socres of Naive Bayes, SVM, and Decision Tree are 0.3, 0.5 and 0.7 respectively. Here I will print all evaluation scores of the three classifiers:

# In[110]:


def print_all_score (classifier, feature = features_new_scaled):
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    score_dict = {}
    for score in scoring:
        score_dict[score] = cross_val_score(
            classifier, feature, y = labels, 
            cv=sss, scoring = score).mean()
    return score_dict


# In[111]:


print_all_score (best_gnb)


# In[112]:


print_all_score (best_svc)


# In[113]:


print_all_score (best_tree)


# As we can see, the 'best_tree' classifier generates the best result. So I will select best_tree as the final model.

# In[97]:


best_tree


# ### Supplement: Test for the Importance of 'from_poi_percent'

# In[102]:


print_all_score(best_tree, features_new_with_from_scaled)


# As we can see, after adding 'from_poi_percent' to the final features. Recall rate of best_tree classifier increases 0.1! So I will add the 'from_poi_percent' to the final features.

# ## 4. Answers to Free-Response Questions  

# ### 4.1 Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those? 

# The goal of this project is to find suspects from all persons related with Enron. 
# The dataset contains three classes of data: 1. The finance data, which contains total payments of each person received and compositions of payments. 2. The stocks data, which contains total stocks of each person and compositions of stocks. 3. The emails data, which contains the number of emails that each person sent and received, and how many of which are from pois. There are 146 data points in the dataset, 18 of them are pois. There are 20 features and 1 label(poi). There are many missing values and I imputed them using 0.
# There are three outliers. One is 'TOTAL', which is the total of all features. One is ''THE TRAVEL AGENCY IN THE PARK', which certainly is not a person. The last is 'LOCKHART EUGENE E', whose features are all missing. I deleted them at the very beginning of my project.

# ### 4.2	What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come readymade in the dataset explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

# I used these 7 features: 'bonus', 'loan_advances', 'total_payments', 'total_stock_value', 'to_poi_percent' , 'from_poi_percent' and 'shared_receipt_with_poi'. I selected them using ‘f_classif’ method which uses ANOVA F-value and p-value to evaluate the linear relation between features and labels. I selected these 6 because their p-values are less than 0.05, which means that they significantly associated with poi. I used SelectKBest method and let n = 6 to select them. After that, I tested whether adding 'from_poi_percent' to the final features will enhance the performance. The test result showed that it worked! So I add 'from_poi_percent' to the final features. 
# I did feature scaling. Because features here have different scales. Some algorithms like SVM do need feature scaling.
# I created my own features: 'from_poi_percent' and 'to_poi_percent'. I did this because it is better to use the proportion of how many emails are from and to pois rather than exact numbers.

# ### 4.3	What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

# I used SVM in the end. I also tried Naïve Bayes and Decision Tree. For the final question, please refer part 3 of my project.

# ### 4.4	What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier). 

# Tuning parameters can make algorithms fit your data more. If you don’t do that, default parameters will be used. The performance and scores will be bad. 
# I used ‘GridSearchCV’ method to tune parameters. I have tuned 'C', 'kernel' and 'gamma' for SVM and 'min_samples_split' for Decision Tree. I did not tune any parameters for Naïve Bayes, because there are nothings to be tuned.

# ### 4.5 What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis? 

# Cross validation is dividing datasets to different training and testing datasets and doing cross validation using them. 
# The classic mistake you can make if you do it wrong is overfitting. 
# I used StratifiedShuffleSplit as the cross-validation generator, becuase: The poi is binary variable, we need to make sure all groups after spliting have both poi and non-poi. And I let n_splits=5, which means there will be 5-fold cross validation. Then I assigned this to sss.
# Finally, I used ‘GridSearchCV’ and ‘cross_val_score’ methods and let cv = sss.

# ### 4.6 Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something humanunderstandable about your algorithm’s performance. 

# I used 'accuracy', 'precision', 'recall' and 'f1' to evaluate the average performance of my models. 
# The accuracy represents the ability of the classifier to find right predictions. The precision represents the ability of the classifier not to label as positive a sample that is negative. The recall represents he ability of the classifier to find all the positive samples. The f1 represents weighted average of the precision and recall. 
# The scores of my final classifier are: 
# {'accuracy': 0.9466666666666667,
#  'f1': 0.6933333333333332,
#  'precision': 0.7333333333333333,
#  'recall': 0.8}

# ## 5. Test

# I recoded your tester.py to make sure it suits my work flow and Python 3.6 and scikit-learn v0.20.0.

# In[114]:


PERF_FORMAT_STRING = "\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, features, labels, folds = 1000):
    cv = StratifiedShuffleSplit(n_splits = folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv.split(features, labels):
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print ("Warning: Found a predicted label not == 0 or 1.")
                print ("All predictions should take value 0 or 1.")
                print ("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print (clf)
        print (PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print (RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print ("")
    except:
        print ("Got a divide by zero when trying out:", clf)
        print ("Precision or recall may be undefined due to a lack of true positive predicitons.")


# In[115]:


test_classifier(best_tree, features_new_with_from_scaled, labels, folds = 1000)

