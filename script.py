# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:23:02 2022

"""

import pandas as pd
import numpy as np
import math

#dezcribe dataset, size
#reverse order: the book popularity, the author popularity, and the age ranges by reading activity

books = pd.read_csv("data\BX-Books.csv", encoding='latin-1', sep=";", error_bad_lines=False)
users = pd.read_csv("data\BX-Users.csv", encoding='latin-1', sep=";", error_bad_lines=False)
ratings = pd.read_csv("data\BX-Book-Ratingstemp.csv", encoding='latin-1', sep=";", error_bad_lines=False)
#
#BX-Book-Ratings.csv
df =  pd.merge(ratings, users)

data = pd.merge(df, books)
data1 = data.head()
print("------------------------------------------------------------------------------")
print('Data Size: ', data.size)# contains info from all the tables
print('Data Shape: ', data.shape)
print('_____Data Describe_____')
print(data.describe())
print('_____Unique Data_____')
print(data.nunique())
#csvframe.info(memory_usage='deep')
print("------------------------------------------------------------------------------")
#Reverse order Book Popularity


def popularity(data):
    dic = {}
    for i in data:
        dic[i] = 0

    for isbn in data:
        if isbn in dic:
            dic[isbn] +=1
    return dic


isbnCounter = popularity(data['ISBN'])
booksReverse_df = pd.DataFrame({'ISBN': isbnCounter.keys(),'nums':  isbnCounter.values()})
#{"ISBN": 2, "I"} 
print("Books with the least popularity")
print(booksReverse_df.sort_values(by='nums', ascending=True))

#Reverse order Author Popularity
authorCounter = popularity(data['Book-Author'])
authorReverse_df = pd.DataFrame({'Book-Author': authorCounter.keys(),'nums':  authorCounter.values()})
print("Authors with the least popularity")
print(authorReverse_df.sort_values(by='nums', ascending=True))

'''
babies = 0-2
children = 3-12
teens = 13-18
young adults = 19-30
Middle-aged adults = 31-45
old Adults = 46-pano
'''
#reading activity


ageRanges = {'0-2':[], '3-12':[], '13-18':[], '19-30':[],'31-45':[], '46>':[], 'null':[]}

for row in range(data.shape[0]):
    age = data['Age'].iloc[[row]]
    isbn = data['ISBN'].iloc[[row]].tolist()
    isbn = isbn[0]
    if math.isnan(age):
        ageRanges['null'].append(isbn)
    elif int(age) <= 2:
        ageRanges['0-2'].append(isbn)
        
    elif int(age) >=3 and int(age)<=12:
        ageRanges['3-12'].append(isbn)
    elif int(age) >=13 and int(age)<=18:
        ageRanges['13-18'].append(isbn)
    elif int(age) >=19 and int(age)<=30:
        ageRanges['19-30'].append(isbn)
    elif int(age) >=31 and int(age)<=45:
        ageRanges['31-45'].append(isbn)
    elif int(age) >= 46:
        ageRanges['46>'].append(isbn)

def rng(x):
    rslt = []
    for i in x:
        rslt.append(len(i))
    return rslt
df1 = pd.DataFrame({'ageRange':ageRanges.keys(), 'ReadingActivity': rng(ageRanges.values())})