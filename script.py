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
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
from scipy.stats import kendalltau
from sklearn.decomposition import TruncatedSVD
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

#Books-Popularity Outliers
num_ratings_books = data.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_ratings_books.rename(columns = {'Book-Rating': 'numRatings'}, inplace=True)
#apply zscore: z = (x-μ)/σ
#num_ratings_books['zscore'] = ( num_ratings_books.Ratings - num_ratings_books.Ratings.mean() ) / num_ratings_books.Ratings.std()
num_ratings_books['zscore'] = zscore(num_ratings_books.numRatings)
booksOutliers = num_ratings_books[(num_ratings_books.zscore > -3) & (num_ratings_books.zscore < 3)]
print("\n\t\t\t\t******Books-Popularity Outliers(sorted) Q1******\n")
print(booksOutliers.sort_values('zscore'))

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


#-----------------------------------OUTLIER--ZSCORE
#books-outlier
ratings_name = ratings.merge(books, on='ISBN')
#plt.style.use('ggplot')
ratings_name['Books-Count'] = ratings_name.groupby('Book-Title')['Book-Title'].transform('count')
ratings_name.hist('Books-Count',bins=30)
ratings_name['z-score-books'] = np.abs(stats.zscore(ratings_name['Books-Count']))
ratings_name = ratings_name[ratings_name['z-score-books'] > 0.48 ]
#ratings_name.hist('Books-Count',bins=4)

#Author-Outlier
ratings_name['Author-count'] = ratings_name.groupby('Book-Author')['Book-Author'].transform('count')
# final_ratings.hist('Author-Count',bins=4)
ratings_name['z-score-Author'] = np.abs(stats.zscore(ratings_name['Author-count']))
ratings_name = ratings_name[ratings_name['z-score-Author'] > 0.3 ]
# final_ratings.hist('Author-Count',bins=4)

#Users-Ouoliers
ratings_name['User-count'] = ratings_name.groupby('User-ID')['User-ID'].transform('count')
# final_ratings.hist('Author-Count',bins=4)
ratings_name['z-score-User'] = np.abs(stats.zscore(ratings_name['User-count']))
ratings_name = ratings_name[ratings_name['z-score-User'] > 0.2 ]
# final_ratings.hist('Author-Count',bins=4)
ratings_name.hist('Books-Count',bins=30)

#____________COLABORATIVE FILTERING______________

#select users who have rated more than 200 books
x = ratings_name.groupby('User-ID').count()['Book-Rating'] >= 68
#x[x] => returns all the true .index gives the users
wellread_users = x[x].index
print('there are only', wellread_users.shape, 'who have read more than 68 books')

#filtering entres from ratings_name only rated by users in wellread_users
filtering_rating = ratings_name[ratings_name['User-ID'].isin(wellread_users)]
#----------
#selectiong the books that have more than 40 ratings 
y = filtering_rating.groupby('Book-Title').count()['Book-Rating'] >=40

famous_books = y[y].index

final_ratings = filtering_rating[filtering_rating['Book-Title'].isin(famous_books)]
#print('finalRating: ', final_ratings.shape)
#making pivot table
book_pivot = final_ratings.pivot_table(index='User-ID', 
                                       columns='Book-Title', 
                                       values='Book-Rating')
#replace nan values
book_pivot.fillna(0,inplace=True)
#-----------------------------------OUTLIER--ZSCORE----------------------------

# kn_df = pd.DataFrame({'User_ID': kn.keys(),'kn': kn.values()})

# #______Find similarities______
# #Extract similarities_df to csv 
# similarities_df.to_csv('user-pairs-books.data', index = False)



#-----------____-------____---______----______---_____--______-

def findKSimilar_users (r, k):
        
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
nNeighbours=7
nUsers = book_pivot.shape[0]
similarUsers, similarities=findKSimilar_users (book_pivot, nNeighbours)




similarUsers_index =  {}
for i, user in enumerate(similarUsers):
    similarUsers_index[i] = user
similarUsers_index_df = pd.DataFrame({'User_ID': similarUsers_index.keys(),'kn': similarUsers_index.values()})
#Extract kn_df to json file    
similarUsers_index_df.to_json('neighbors-k-books.json', orient = 'split', index = False)   
similarities_df = pd.DataFrame(similarities)
similarities_df.to_csv('user-pairs-books.data', index = False)




#//similarities, similarUsers
#-----------____-------____---______----______---_____--______-
# similarItems, similarities=findKSimilar (book_pivot.T, nNeighbours)
#recommened books for the active user
# final_ratings['User-ID'] == '278418'
# book_pivot.index[active_similarUsers]

# book_pivot[book_pivot.index == '100459']

# similarity_scores = cosine_similarity(book_pivot.T)
#user, book brings all the books of the user
# book_pivot.iloc[138,:]
def get_recommended_books(active_user): 
    active_similarUsers = similarUsers[active_user]
    active_similarUsers = active_similarUsers.astype(int)
    ids = book_pivot.index[active_similarUsers]
    data = []
    for _id in ids:
        temp = final_ratings[final_ratings['User-ID'] == _id][['Book-Title', 'Book-Rating' ]]
        book = temp[temp['Book-Rating'] == temp['Book-Rating'].max()]['Book-Title'].values[0]
        # book = list(book)
        data.append(book)
    data = set(data)
    data = list(data)
    for i, b in enumerate(data):
        print(i, b)
# for activeUser in active_similarUsers:
    



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
        sum= sum+ data[itemId,neighbor]*similarities[userId,neighbor]
        simSum = simSum + similarities[userId,neighbor]
    if simSum <=0:
        return 0 
    else:
        return  (sum/simSum)

book_array = np.array(book_pivot.T)
hideUserID = 10
hideItemID = 2
#prediction=predict(hideUserID,hideItemID,book_array, similarUsers, similarities)
#print ('prediction:',prediction, 'real:',book_pivot.iloc[hideUserID,hideItemID])

matrix = pd.DataFrame(book_pivot)
def get_recommended_books(active_user, k):
    predictions_book = {}
    
    for i in range(0,book_pivot.shape[1]):
        predictions_book[matrix.columns[i]] = predict(active_user,i,book_array, similarUsers, similarities)
    
    list_p = sorted(predictions_book.items(), key=lambda x:x[1], reverse = True)
    for j in range(0,k):
        print(j+1, list_p[j][0], float(predictions_book[list_p[j][0]]))



active_user = 100

get_recommended_books(active_user, 5)    
                

def maeRmsePrecisionRecall(r, similarUsers, similarities):
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
mae, rmse, precision, recall = maeRmsePrecisionRecall(book_array, similarUsers, similarities)

if precision !=0 and recall !=0:
    f1=2*precision*recall/(precision+recall)
    print ('\nF1=',f1)


#------------------------------------Matrix Factorization Recommender-----------------------------------------
print('\n\t\t\t------------------Matrix Factorization------------------')
def mf_find_similar_users(matrix, k):
    
    mf_similarUsers = -1*np.ones((nUsers, k))
    SVD=TruncatedSVD(n_components=12, random_state=17)
    matrixx = SVD.fit_transform(matrix)
    mf_corr = corr=np.corrcoef(matrixx)
    
    for i in range(0, nUsers):
        simUsersIdxs= np.argsort(mf_corr[:,i])
        
        l=0
        #find its most similar users    
        for j in range(simUsersIdxs.size-2, simUsersIdxs.size-k-2,-1):
            simUsersIdxs[-k+1:]
            mf_similarUsers[i,l]=simUsersIdxs[j]
            l=l+1
            
    return mf_similarUsers, mf_corr


mf_similarUsers, mf_corr = mf_find_similar_users(book_pivot, 7)


#so we can find the sum we fill the array with 0
mf_corr = np.nan_to_num(mf_corr)

mf_mae, mf_rmse, mf_precision, mf_recall = maeRmsePrecisionRecall(book_array, mf_similarUsers, mf_corr)

if mf_precision !=0 and mf_recall !=0:
    mf_f1=2*mf_precision*mf_recall/(mf_precision+mf_recall)
    print ('\nF1=',mf_f1)


matrix = pd.DataFrame(book_pivot)
def mf_get_recommended_books(active_user, k):
    predictions_book = {}
    
    for i in range(0,book_pivot.shape[1]):
        predictions_book[matrix.columns[i]] = predict(active_user,i,book_array, mf_similarUsers, mf_corr)
    
    list_p = sorted(predictions_book.items(), key=lambda x:x[1], reverse = True)
    for j in range(0,k):
        print(j+1, list_p[j][0],',', float(predictions_book[list_p[j][0]]))

active_user = 100

mf_get_recommended_books(active_user, 5)    


