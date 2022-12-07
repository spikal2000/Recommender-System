# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 19:34:40 2022

@author: spika
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

from sklearn.neighbors import NearestNeighbors

class recommend_books:
    #self.nUsers = 0
    def __init__(self):
        self.nNeighbours = 6
        self.recommend()
        #self.get_recommended_books()
        # self.self.nUsers = self.nUsers
        # nNeighbours=7
    
        #function that returns similar users and similarities arrays
    def findKSimilar (self,r):
            
        # similarUsers is 2-D matrix
        k = self.nNeighbours
        nUsers = self.nUsers
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
        
    
    
    def recommend(self):
        #import the csvs
        books = pd.read_csv("data\BX-Books.csv", encoding='ISO-8859-1', sep=";", on_bad_lines='skip', 
        dtype={'ISBN': 'str', 'Book-Title': 'str', "Book-Author": 'str',"Year-Of-Publication": 'str',"Publisher": 'str',"Image-URL-S": 'str',"Image-URL-M": 'str',"Image-URL-L": 'str'})
        users = pd.read_csv("data\BX-Users.csv", encoding='ISO-8859-1', sep=";", on_bad_lines='skip', 
                            dtype={"User-ID": 'str', "Location": 'str', "Age": 'str'})
        ratings = pd.read_csv("data\BX-Book-Ratings.csv", encoding='ISO-8859-1', sep=";", on_bad_lines='skip',
                              dtype={"User-ID": 'str',"ISBN":'str',"Book-Rating":'int'})
        
        #TODO: describe the data and plots-----------------------------------
        
        #-----------------------------------OUTLIER--ZSCORE-----------------------------------
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
        
        #----------------Remove some users and create Pivot table for books
        #select users who have rated more than 68 books
        x = ratings_name.groupby('User-ID').count()['Book-Rating'] >= 68
        #x[x] => returns all the true .index gives the users
        wellread_users = x[x].index
        #print('there are only', wellread_users.shape, 'who have read more than 68 books')

        #filtering entres from ratings_name only rated by users in wellread_users
        filtering_rating = ratings_name[ratings_name['User-ID'].isin(wellread_users)]
        #----------
        #selectiong the books that have more than 40 ratings 
        y = filtering_rating.groupby('Book-Title').count()['Book-Rating'] >=49

        famous_books = y[y].index

        final_ratings = filtering_rating[filtering_rating['Book-Title'].isin(famous_books)]
        #print('finalRating: ', final_ratings.shape)
        #making pivot table
        book_pivot = final_ratings.pivot_table(index='User-ID', 
                                               columns='Book-Title', 
                                               values='Book-Rating')
        #replace nan values
        book_pivot.fillna(0,inplace=True)
        self.book_pivot = book_pivot
        
        #-----------____-------____---______----______---_____--______-
        self.nUsers = book_pivot.shape[0]
        
        self.similarUsers, self.similarities=self.findKSimilar(book_pivot)
        
        
        
        
        
        
    
    def export_data(self):
        similarUsers_index =  {}
        for i, user in enumerate(self.similarUsers):
            similarUsers_index[i] = user
        similarUsers_index_df = pd.DataFrame({'User_ID': similarUsers_index.keys(),'kn': similarUsers_index.values()})
        #Extract kn_df to json file    
        similarUsers_index_df.to_json('neighbors-k-books.json', orient = 'split', index = False)   
        similarities_df = pd.DataFrame(self.similarities)
        similarities_df.to_csv('user-pairs-books.data', index = False)
        
    
        
         
    def get_recommended_books(self, active_user):
        similarUsers = self.similarUsers
        book_pivot = self.book_pivot
        active_similarUsers = similarUsers[active_user]
        active_similarUsers = active_similarUsers.astype(int)
        data = []
        for k in range(len(book_pivot.index[active_similarUsers])):
            data.append(book_pivot.columns[active_similarUsers][k])
            
        print('\nThe recommened books for the user', active_user, "are:\n")
        for i, book in enumerate(data):
            print(i+1, book)
        
        
        
        
        
        
        