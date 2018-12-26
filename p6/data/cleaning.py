# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 22:46:22 2018

@author: Tianxing Zhai
"""

import pandas as pd
import os
os.chdir('E:\\xampp\\htdocs\\projects\\titanic\\data')


data = pd.read_csv('titanic_data.csv')

data.loc[data['Survived'] == 1, 'Survived'] = 'Survived'
data.loc[data['Survived'] == 0, 'Survived'] = 'Died'

data.loc[data['Pclass'] == 1, 'Pclass'] = '1st class'
data.loc[data['Pclass'] == 2, 'Pclass'] = '2nd class'
data.loc[data['Pclass'] == 3, 'Pclass'] = '3rd class'

data['Age_cat'] = pd.cut(data['Age'], [0, 18, 50, 80], labels=['Children', 'Adults','Seniors'])
data['Age_cat'] = data['Age_cat'].astype('object')

data.loc[data['Embarked'] == 'C', 'Embarked'] = 'Cherbourg'
data.loc[data['Embarked'] == 'Q', 'Embarked'] = 'Queenstown'
data.loc[data['Embarked'] == 'S', 'Embarked'] = 'Southampton'

data_keep = data[['Survived', 'Pclass', 'Sex', 'Age_cat', 'Embarked']]
data_keep = data_keep.dropna()
data_keep['count'] = 1


data_keep.to_csv('titanic_data_cleaned.csv')
