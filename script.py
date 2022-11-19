# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:23:02 2022

"""
import re
import pandas as pd
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
#dezcribe dataset, size
#reverse order: the book popularity, the author popularity, and the age ranges by reading activity

books = pd.read_csv("data\BX-Books.csv", encoding='ISO-8859-1', sep=";", on_bad_lines='skip', 
dtype={'ISBN': 'str', 'Book-Title': 'str', "Book-Author": 'str',"Year-Of-Publication": 'str',"Publisher": 'str',"Image-URL-S": 'str',"Image-URL-M": 'str',"Image-URL-L": 'str'})
users = pd.read_csv("data\BX-Users.csv", encoding='ISO-8859-1', sep=";", on_bad_lines='skip', 
                    dtype={"User-ID": 'str', "Location": 'str', "Age": 'str'})
ratings = pd.read_csv("data\BX-Book-Ratings.csv", encoding='ISO-8859-1', sep=";", on_bad_lines='skip',
                      dtype={"User-ID": 'str',"ISBN":'str',"Book-Rating":'str'})

#BX-Book-Ratings.csv
df =  pd.merge(ratings, users)

data = pd.merge(df, books)
#Drop all nan values
data.dropna(inplace=True)
#delete random book values '~' => remove
#data['Book-Title'].loc[~data['Book-Title'].str.contains('\?', flags=re.I, regex=True)]
#data.drop(data[data['Book-Title'].str.contains('?')], inplace = True)
#convert Ages to int
data['Age'] = data['Age'].astype('int')
#data1 = data.head()
print("\t\t---------------------------------DataInfo---------------------------------")
print('Data Size: ', data.size)# contains info from all the tables
print('Data Shape: ', data.shape)
print('_____Data Describe_____')
print(data.describe())
print('_____Number of Books_____')
print(data['Book-Author'].nunique())
print('_____Number of Readers_____')
print(data["User-ID"].nunique())
#csvframe.info(memory_usage='deep')
print("\t\t---------------------------------Q1---------------------------------")
print("\n\t\t\t\t******Reverse order Book Popularity Q1-a******\n")
print(data["Book-Title"].value_counts().sort_values())
print("\n\t\t\t\t******Reverse order Author Popularity Q1-b******\n")
print(data["Book-Author"].value_counts().sort_values())

#babies = 0-2
#children = 3-12
#teens = 13-18
#young adults = 19-30
#Middle-aged adults = 31-45
#old Adults = 46-pano

print("\n\t\t\t\t******Age ranges by reading activity Q1-c******\n")

ageRanges = pd.DataFrame({'Age': data['Age']})
ranges = [0, 2, 12, 18, 30, 45, 500]
ageRanges = ageRanges.groupby(pd.cut(ageRanges.Age, ranges)).count()
print(ageRanges)

#----------------------------------------------------------------------------------------------------------------------------------------
print("\t\t------------------------------Q1_Outlier detection I------------------------------")

#books- popularity TODO: FIX HIST
booksPopularity = pd.DataFrame(data["Book-Title"].value_counts())
booksPopularity.rename(columns = {'Book-Title': 'Book-Popularity'}, inplace = True)
booksPopularity.reset_index(inplace = True)
plt.hist(booksPopularity['Book-Popularity'])


usersPopularity = pd.DataFrame('Users': data['User-ID', ])







