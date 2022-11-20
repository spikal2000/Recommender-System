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
                      dtype={"User-ID": 'str',"ISBN":'str',"Book-Rating":'int'})

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

def zscore(nums):
    z = ( nums - nums.mean() ) / nums.std()
    return z


#Books-Popularity Outliers
num_ratings_books = data.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_ratings_books.rename(columns = {'Book-Rating': 'Ratings'}, inplace=True)
#apply zscore: z = (x-μ)/σ
#num_ratings_books['zscore'] = ( num_ratings_books.Ratings - num_ratings_books.Ratings.mean() ) / num_ratings_books.Ratings.std()
num_ratings_books['zscore'] = zscore(num_ratings_books.Ratings)
booksOutliers = num_ratings_books[(num_ratings_books.zscore > -3) & (num_ratings_books.zscore < 3)]
print("\n\t\t\t\t******Books-Popularity Outliers(sorted) Q1******\n")
print(booksOutliers.sort_values('zscore'))


#Authors-Popularity Outliers
num_ratings_authors = data.groupby('Book-Author').count()['Book-Rating'].reset_index()
num_ratings_authors.rename(columns = {'Book-Rating': 'Ratings'}, inplace=True)
#apply zscore: z = (x-μ)/σ
#num_ratings_authors['zscore'] = ( num_ratings_authors.Ratings - num_ratings_authors.Ratings.mean() ) / num_ratings_authors.Ratings.std()
num_ratings_authors['zscore'] = zscore(num_ratings_authors.Ratings)
authorOutliers = num_ratings_authors[(num_ratings_authors.zscore > -3) & (num_ratings_authors.zscore < 3)]
print("\n\t\t\t\t******Authors-Popularity Outliers(sorted) Q1******\n")
print(authorOutliers.sort_values('zscore'))


#





#usersPopularity = pd.DataFrame('Users': data['User-ID' ])
#numRating = data.groupby('Book-Title').count()['Book-Rating'].reset_index()
#numRating.rename(columns={'Book-Rating': 'Num-Book-Rating'}, inplace=True)


'''
#numRating['Avg_Num_Rating'] =
# 1.book title, number of ratings
num_ratings = data.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_ratings.rename(columns = {'Book-Rating': 'Ratings'}, inplace=True)

avg_ratings = data.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_ratings.rename(columns = {'Book-Rating': 'AvgRatings'}, inplace=True)

popular_Book = num_ratings.merge(avg_ratings, on='Book-Title')
'''
#popular_Book['zscore'] = ( popular_Book.AvgRatings - popular_Book.AvgRatings.mean() ) / popular_Book.AvgRatings.std()


#popular_Book['zscore'] = popular_Book['zscore'] ** 2

#popular_Book_no_outliers = popular_Book[(popular_Book.zscore > -3) & (popular_Book.zscore < 3)]
#num_ratings[(num_ratings.zscore > 2) & (num_ratings.zscore <50)].sort_values('zscore').tail(30)