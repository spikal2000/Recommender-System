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
from sklearn.metrics.pairwise import cosine_similarity
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
data = data.head(100000)
#Drop all nan values
data.dropna(inplace=True)
#delete random book values '~' => remove
#data['Book-Title'].loc[~data['Book-Title'].str.contains('\?', flags=re.I, regex=True)]
#data.drop(data[data['Book-Title'].str.contains('?')], inplace = True)
#convert Ages to int
data['Age'] = data['Age'].astype('int')
data = data.reset_index()
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
#data['Book-Rating'].unique()
#Out[59]: array([ 5,  0,  8,  9,  6,  7,  4,  3, 10,  1,  2])
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
print("\t\t--------------------------Q1_Outlier detection I--------------------------")

#Function to calculate zscore
def zscore(nums):
    z = ( nums - nums.mean() ) / nums.std()
    return z


#Books-Popularity Outliers
num_ratings_books = data.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_ratings_books.rename(columns = {'Book-Rating': 'numRatings'}, inplace=True)
#apply zscore: z = (x-μ)/σ
#num_ratings_books['zscore'] = ( num_ratings_books.Ratings - num_ratings_books.Ratings.mean() ) / num_ratings_books.Ratings.std()
num_ratings_books['zscore'] = zscore(num_ratings_books.numRatings)
booksOutliers = num_ratings_books[(num_ratings_books.zscore > -3) & (num_ratings_books.zscore < 3)]
print("\n\t\t\t\t******Books-Popularity Outliers(sorted) Q1******\n")
print(booksOutliers.sort_values('zscore'))


#Authors-Popularity Outliers
num_ratings_authors = data.groupby('Book-Author').count()['Book-Rating'].reset_index()
num_ratings_authors.rename(columns = {'Book-Rating': 'numRatings'}, inplace=True)
#apply zscore: z = (x-μ)/σ
#num_ratings_authors['zscore'] = ( num_ratings_authors.Ratings - num_ratings_authors.Ratings.mean() ) / num_ratings_authors.Ratings.std()
num_ratings_authors['zscore'] = zscore(num_ratings_authors.numRatings)
authorOutliers = num_ratings_authors[(num_ratings_authors.zscore > -3) & (num_ratings_authors.zscore < 3)]
print("\n\t\t\t\t******Authors-Popularity Outliers(sorted) Q1******\n")
print(authorOutliers.sort_values('zscore'))


#USer-IDs Books Outliers
num_ratings_user = data.groupby('User-ID').count()['ISBN'].reset_index()
num_ratings_user.rename(columns = {'ISBN': 'numISBN'}, inplace=True)
#apply zscore: z = (x-μ)/σ
num_ratings_user['zscore'] = zscore(num_ratings_user.numISBN)
userOutliers = num_ratings_user[(num_ratings_user.zscore > -3) & (num_ratings_user.zscore < 3)]
print("\n\t\t\t\t******Users-Number of ISBN Outliers(sorted) Q1******\n")
print(userOutliers.sort_values('zscore'))

#-----------------------------------------------------------------------------------------------------------------------
print("\t\t--------------------------Q2_Recommender System--------------------------")
print("\n\t\t\t\t******Find similarities Q2-a******\n")

#{'User-ID': [books]}
'''
user_books = {}

for i in range(1, len(data)):
    user = data['User-ID'][i]
    if (userOutliers.loc[userOutliers['User-ID'] == user]['User-ID'] is user):
        print((userOutliers.loc[userOutliers['User-ID'] == user]['User-ID'] is user))
        if user not in user_books:
            user_books[user] = [data['ISBN'][i]]
        else:
            user_books[user].append(data['ISBN'][i])
            
            
            ###userOutliers[userOutliers['zscore']>-3]
'''

#Similarity

ratings_books = ratings.merge(books, on='ISBN')

number_rating = ratings_books.groupby('Book-Title')['Book-Rating'].count().reset_index()

number_rating.rename(columns= {'Book-Rating':'number_of_ratings'}, inplace=True)

final_rating = ratings_books.merge(number_rating, on='Book-Title')

final_rating = final_rating[final_rating['number_of_ratings'] >= 100]

final_rating.drop_duplicates(['User-ID','Book-Title'], inplace=True)


book_pivot = final_rating.pivot_table(columns='Book-Title', 
                                       index='User-ID',
                                       values='Book-Rating')
book_pivot.fillna(0, inplace=True)
#book_pivot = book_pivot.T

#for user in range(0, book_pivot.shape[0]):
#    for book in range(0,book_pivot.shape[1]):
#        print(book_pivot.iloc[ user, col])



def findKSimilar(r, k):
    
    # similarUsers is 2-D matrix
    similarUsers=-1*np.ones((nUsers,k))
    
    similarities=cosine_similarity(r)
       
    # for each user
    for i in range(0, nUsers):
        simUsersIdxs= np.argsort(similarities[:,i])
        
        l=0
        #find its most similar users    
        for j in range(simUsersIdxs.size-2, simUsersIdxs.size-k-2,-1):
            simUsersIdxs[-k+1:]
            similarUsers[i,l]=simUsersIdxs[j]
            l=l+1
            
    return similarUsers, similarities

nNeighbours=2
nUsers = len(book_pivot.iloc[ 0, :])

similarUsers, similarities=findKSimilar(book_pivot.T, nNeighbours)






'''
ratings_matrix = pd.pivot_table(ratings, values=["Book-Rating"], index=['User-ID', 'ISBN'])


#ratings_matrix.to_csv('user-pairs-books.data')

#userId = ratings_matrix.index

#ISBN = ratings_matrix.columns
#ratings_matrix.shape

def findKSimilar(r, k):
    
    # similarUsers is 2-D matrix
    similarUsers=-1*np.ones((nUsers,k))
    
    similarities=cosine_similarity(r)
       
    # for each user
    for i in range(0, nUsers):
        simUsersIdxs= np.argsort(similarities[:,i])
        
        l=0
        #find its most similar users    
        for j in range(simUsersIdxs.size-2, simUsersIdxs.size-k-2,-1):
            simUsersIdxs[-k+1:]
            similarUsers[i,l]=simUsersIdxs[j]
            l=l+1
            
    return similarUsers, similarities

#ratings_matrix = ratings_matrix.fillna(0)
nNeighbours=2
nUsers = len(ratings_matrix.index)
similarUsers, similarities=findKSimilar(ratings_matrix, nNeighbours)

ratings_matrix

#[ 96.,  94.]
'''








