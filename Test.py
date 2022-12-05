# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 15:55:34 2022

@author: spika
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:23:02 2022

"""
import numpy as np
import pandas as pd
#dezcribe dataset, size
#reverse order: the book popularity, the author popularity, and the age ranges by reading activity

books = pd.read_csv("data\BX-Books-TEMP.csv", encoding='ISO-8859-1', sep=";", on_bad_lines='skip')
#dtype={'ISBN': 'str', 'Book-Title': 'str', "Book-Author": 'str',"Year-Of-Publication": 'str',"Publisher": 'str',"Image-URL-S": 'str',"Image-URL-M": 'str',"Image-URL-L": 'str'})
users = pd.read_csv("data\BX-Users-TEMP.csv", encoding='ISO-8859-1', sep=";", on_bad_lines='skip')
                    #dtype={"User-ID": 'str', "Location": 'str', "Age": 'str'})
ratings = pd.read_csv("data\BX-Book-Ratings-TEMP.csv", encoding='ISO-8859-1', sep=";", on_bad_lines='skip')
                      #dtype={"User-ID": 'str',"ISBN":'str',"Book-Rating":'int'})

#Checking the dataset for null and duplicates

#books, users, ratings that have null
print(books.isnull().sum())
print(users.isnull().sum())
print(ratings.isnull().sum())

#checking duplicated b=values in the dataset 
print(books.duplicated().sum())
print(users.duplicated().sum())
print(ratings.duplicated().sum())

#Data analysis
print('we have', books['Book-Title'].count(), 'books in our dataset')
print('and', books['Book-Title'].nunique(), 'of them are unique')


#Checking the number of occurances for all the books 
book_count = pd.DataFrame(books['Book-Title'].value_counts().reset_index())

book_count.rename(columns={'index':'Book-Title', 'Book-Title': 'count'}, inplace=True)
print('as we can see we have multiple entrys for the same book')
print(book_count)

#we drop duplicated entries using the book-title column and only select rows with unique Book-titles
user_rating_count = pd.DataFrame(ratings['User-ID'].value_counts().reset_index())

user_rating_count.rename(columns={'index':'User-ID', 'User-ID': 'count'}, inplace=True)
# a small number of users have rated a large amount of books

ratings_name = ratings.merge(books, on='ISBN')


#____________COLABORATIVE FILTERING______________
'''
#select users who have rated more than 200 books
x = ratings_name.groupby('User-ID').count()['Book-Rating'] > 200
#x[x] => returns all the true .index gives the users
wellread_users = x[x].index
print('there are only', wellread_users.shape, 'who have read more than 200 books')

#filtering entres from ratings_name only rated by users in wellread_users
filtering_rating = ratings_name[ratings_name['User-ID'].isin(wellread_users)]
#----------
#selectiong the books that have more than 40 ratings 
y = filtering_rating.groupby('Book-Title').count()['Book-Rating']>=40

famous_books = y[y].index

final_ratings = filtering_rating[filtering_rating['Book-Title'].isin(famous_books)]

#making pivot table
book_pivot = final_ratings.pivot_table(index='Book-Title', 
                                       columns='User-ID', 
                                       values='Book-Rating')
#replace nan values
book_pivot.fillna(0,inplace=True)
'''
books_names = ratings.merge(books, on='ISBN')
book_pivot = books_names.pivot_table(index='User-ID', 
                                       columns='Book-Title', 
                                       values='Book-Rating')
book_pivot.fillna(0,inplace=True)
#SIMILARITIES
from sklearn.metrics.pairwise import cosine_similarity

#similarity_score = cosine_similarity(book_pivot)
def findKSimilar (book, k):
    nUsers = book_pivot.shape[0]
    # similarUsers is 2-D matrix
    similarUsers=-1*np.ones((nUsers,k))
    
    similarities=cosine_similarity(book)
       
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
book_array = np.array(book_pivot)
nNeighbours = 2
similarUsers, similarities=findKSimilar (book_array, nNeighbours)
'''
book_similarities = pd.DataFrame(similarities, index=book_pivot.index, columns=book_pivot.index)
def get_similar_movies(movie_name, user_rating):
   
    similar_score = book_similarities[movie_name] * user_rating
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

use1 = [('24 Hours', 2), ('1984', 8)]
similar_scores = pd.DataFrame()
for movie, rating in use1:
    similar_scores = similar_scores.append(get_similar_movies(movie, rating), ignore_index=True)

similar_scores.sum().sort_values(ascending=False).head(10)
'''

'''
#function that recommends books
def recommend(book_name):
    
    #index fetch
    kn = 6
    index = np.where(book_pivot.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarities[index])), key=lambda x:x[1], reverse=True )[1:kn]
    
    data = []
    for i in similar_items:
        item = []
        #books where booktitles are equal to the name of similaritem
        temp_df = books[books['Book-Title'] == book_pivot.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        
        
        data.append(item)
        
    return(data)
 ''' 
 
'''
#index fetch
kn = 6
user_id = '24 Hours'
index = np.where(book_pivot.index == book_name)[0][0]
similar_items = sorted(list(enumerate(similarities[index])), key=lambda x:x[1], reverse=True )[1:kn]
#array of float
similar_users = similarUsers[index] 
data = []
for i in similar_items:
    item = []
    #books where booktitles are equal to the name of similaritem
    temp_df = books[books['Book-Title'] == book_pivot.index[i[0]]]
    item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
    item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
    
    
    data.append(item)
'''   
    
def predict(userId, itemId, data,similarUsers,similarities):

    # number of neighbours to consider
    nCols=similarUsers.shape[1]
    #print(similarUsers.shape[1])
    
    sum=0.0;
    simSum=0.0;
    for l in range(0,nCols):    
        neighbor=int(similarUsers[userId, l])
        #weighted sum
        sum= sum+ data[neighbor,itemId]*similarities[neighbor,userId]
        simSum = simSum + similarities[neighbor,userId]
    
    return  sum/simSum

book_array = np.array(book_pivot)
#hide smthing from book array
prediction=predict(0,2,book_array, similarUsers, similarities)
print ('prediction, real',prediction, book_array[0,2])

#to find the books to recommend
user_to_recommend = 0
#number of books to recommend
recom_books = 2
data={}
for i in range(0,similarUsers.shape[1]):
    
    #temp.append(sorted(enumerate(similarities[int(similarUsers[user_to_recommend][i])]), reverse=True)[1:recom_books+1])
    user = int(similarUsers[user_to_recommend][i])
    print(user)
    data[user] = (sorted(list(similarities[user]), reverse=True)[1:recom_books+1])
    
data_final=[]
for i in data:
    item = []
    #books where booktitles are equal to the name of similaritem
    temp_df = books[books['Book-Title'] == book_pivot.index[i[0]]]
    item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
    item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))

    data_final.append(item)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    