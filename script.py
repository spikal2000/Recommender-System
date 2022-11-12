# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:23:02 2022

"""

import pandas as pd


#dezcribe dataset, size
#reverse order: the book popularity, the author popularity, and the age ranges by reading activity

books = pd.read_csv("data\BX-Books.csv", encoding='latin-1', sep=";", error_bad_lines=False)
users = pd.read_csv("data\BX-Users.csv", encoding='latin-1', sep=";", error_bad_lines=False)
ratings = pd.read_csv("data\BX-Book-Ratings.csv", encoding='latin-1', sep=";", error_bad_lines=False)


