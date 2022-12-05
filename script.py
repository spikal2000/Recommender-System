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

from sklearn.neighbors import NearestNeighbors
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
'''
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
'''
#-----------------------------------------------------------------------------------------------------------------------
print("\t\t--------------------------Q2_Recommender System--------------------------")
print("\n\t\t\t\t******Find similarities Q2-a******\n")

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
book_pivot = final_ratings.pivot_table(index='User-ID', 
                                       columns='Book-Title', 
                                       values='Book-Rating')
#replace nan values
book_pivot.fillna(0,inplace=True)



'''
#_________________________OLd way 
#Similarity

ratings_books = ratings.merge(books, on='ISBN')

number_rating = ratings_books.groupby('Book-Title')['Book-Rating'].count().reset_index()

number_rating.rename(columns= {'Book-Rating':'number_of_ratings'}, inplace=True)

final_rating = ratings_books.merge(number_rating, on='Book-Title')

final_rating = final_rating[final_rating['number_of_ratings'] >= 100]

final_rating.drop_duplicates(['User-ID','Book-Title'], inplace=True)

#final_rating_df = pd.DataFrame(np.random.randn(8, 4),index=[User-ID], columns=['A', 'B', 'C', 'D'])


book_pivot = final_rating.pivot_table(columns='User-ID', 
                                       index='Book-Title',
                                       values='Book-Rating')
book_pivot.fillna(0, inplace=True)
#book_pivot = book_pivot
'''
#___________________________________


#________________Recomender_______________
'''
from scipy.sparse import csr_matrix
book_sparse=csr_matrix(book_pivot)


from sklearn.neighbors import NearestNeighbors
model=NearestNeighbors(metric = 'cosine', algorithm='brute', n_neighbors=3) ## model

model.fit(book_sparse)

#book_pivot.iloc[237,:].values.reshape(1,-1)

'''
'''
#IMPORTANT CHECK THIS FOR SUGGESTIONS
User = 54

#recommended shit TODO: def to call 
#distances => similarities, suggestions => kn
distances,suggestions=model.kneighbors(book_pivot.iloc[User,:].values.reshape(1,-1))
for i in range(len(suggestions)):
    for k in range(len(book_pivot.index[suggestions[i]])):
        print(book_pivot.index[suggestions[i]][k])
'''
'''

#find kn, similarities
userids = book_pivot.columns.tolist()
sim = []
kn = {}
for user in range(0, book_pivot.shape[1]):
    distances,suggestions=model.kneighbors(book_pivot.iloc[user,:].values.reshape(1,-1))
    for l in distances:
        sim.append(l.tolist())
    for s in suggestions:
        for user in userids:
           if user in kn:
               continue
           else:
               kn[user] = s.tolist()
               break
    
similarities_df = pd.DataFrame({'similarities': sim})


kn_df = pd.DataFrame({'User_ID': kn.keys(),'kn': kn.values()})

#______Find similarities______
#Extract similarities_df to csv 
similarities_df.to_csv('user-pairs-books.data', index = False)
#Extract kn_df to json file    
kn_df.to_json('neighbors-k-books.json', orient = 'split', index = False)

'''
#-----------____-------____---______----______---_____--______-

def findKSimilar (r, k):
        
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
nNeighbours=5
nUsers = book_pivot.shape[0]
similarUsers, similarities=findKSimilar (book_pivot, nNeighbours)
#//similarities, similarUsers
#-----------____-------____---______----______---_____--______-

#recommened books for the active user

def get_recommended_books(active_user):
    active_similarUsers = similarUsers[active_user]
    active_similarUsers = active_similarUsers.astype(int)
    data = []
    for k in range(len(book_pivot.columns[active_similarUsers])):
        data.append(book_pivot.columns[active_similarUsers][k])
        
    print('\nThe recommened books for the user', active_user, "are:\n")
    for i, book in enumerate(data):
        print(i+1, book)

active_user = 54
get_recommended_books(active_user)

#-----------------------------------------------------/recomendersystem
#-------------------------RMSE
#print('\n------------------prediction time btcs------------------')
def predict(userId, itemId, data,similarUsers,similarities):

    # number of neighbours to consider
    nCols=similarUsers.shape[1]
    #print(similarUsers.shape[1])
    
    sum=0.0;
    simSum=0.0;
    for l in range(0,nCols):    
        neighbor=int(similarUsers[userId, l])
        #weighted sum
        sum= sum+ data[itemId,neighbor]*similarities[neighbor,userId]
        simSum = simSum + similarities[neighbor,userId]
        if simSum != 0:
            return sum/simSum
        else:
            return 0 
    #return  sum/simSum

book_array = np.array(book_pivot.T)
hideUserID = 10
hideItemID = 2
#prediction=predict(hideUserID,hideItemID,book_array, similarUsers, similarities)
#print ('prediction:',prediction, 'real:',book_pivot.iloc[hideUserID,hideItemID])

def maeRmse(r, similarUsers, similarities):
    predicted_values = []
    tp=fn=fp=fn=0
    for userId in range(0,r.shape[1]):
        for itemId in range(0,r.shape[0]):
            predicted_values.append(predict(userId,itemId,r, similarUsers, similarities))
            #predict recall
            rhat = predict(userId,itemId,r, similarUsers, similarities)
            #print(j, rhat)
            if rhat>=3 and r[itemId,userId]>=3:
                tp=tp+1
            elif rhat>=3 and r[itemId,userId]<3:
                fp = fp+1
            elif rhat<3 and r[itemId,userId]>=3:
                fn=fn+1
                
    predicted_values =  pd.Series(predicted_values, dtype=object).fillna(0).tolist()
    realValues = r.flatten()
    mae = 0
    rmse = 0
    for i in range(0,len(realValues)):
        mae += abs((predicted_values[i] - realValues[i]))
        rmse += (predicted_values[i] - realValues[i]) ** 2

    mae = mae/len(realValues)
    rmse = rmse/len(realValues)
    precision=tp/(tp+fp)
    recall = tp/(tp+fn)
    return mae, rmse, precision, recall

#get mae, rmse, precision, recall
mae, rmse, precision, recall = maeRmse(book_array, similarUsers, similarities)

if precision !=0 and recall !=0:
    f1=2*precision*recall/(precision+recall)
    print ('\nF1=',f1)














