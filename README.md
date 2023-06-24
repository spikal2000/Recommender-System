# Recommender System
#overview
The project analyzes a book dataset to explore various aspects like book popularity, author popularity, reading activity by age, and outlier detection.
A hybrid recommendation system is developed by combining user-based collaborative filtering and matrix factorization techniques. This system generates personalized book recommendations for users based on their previous interactions and similarity to other users.
The evaluation of the recommendation system is done using precision, recall, and F1 score.

 
In order to run the code you need to have a folder called data and the 3 data sets in it.
i)BX-Books.csv
ii)BX-Users.csv
iii)BX-Book-Ratings.csv

# Matrix_Factorizarion
Run the following method in order to get a diffrent user recommendations:
mf_get_recommended_books(active_user, 5) ex: mf_get_recommended_books(105, 5)
NOte: there are only 161 users (it is not based on User-ID)

# Recommender
Run the following method in order to get a diffrent user recommendations:
get_recommended_books(active_user,5) ex: get_recommended_books(105, 5)
NOte: there are only 161 users (it is not based on User-ID)

# github: https://github.com/spikal2000/Recommender-System
