# RECOMMENDATION SYSTEMS

This repository contains Python code files for model development for each recommendation model type and three recommendation system projects.

***Note: The codes are from the Miuul 'Recommendation Systems' course. Some parts of the codes are modified by me.***

- Clone the 'Recommendation Systems' repository using your terminal or git bash.

```
git clone https://github.com/burcukuleci/Recommendation_Systems.git
```
- Download all required packages using the requirements.txt file by running the below command in the terminal.

```
pip install -r requirements.txt
```

- All required data files (except 'rating.csv' and 'armut.csv') are in *datasets* directory. 

***Note: This README.md file provides short information for each Python file. Separate markdown files explain the code and the project in detail. Please refer to those markdown files for detailed information.***



## ASSOCIATION RULE LEARNING

Products are recommended by the rules that are extracted by association rule learning. 

The Apriori algorithm is a basket analysis method that reveals product associations.

Parameters are that used for generating association rules are:

- **N**: Total number of baskets/transactions.

- **Support**(X, Y) = Freq(X, Y) / N  : Probability of X and Y itemsets to be present in a purchase/invoice. (X, Y) association occurs support(%) of the transactions.

- **Confidence**(X, Y) =  Freq(X, Y) / Freq(X) : If X is purchased, there is a confidence(%) chance to purchase Y.

- **Lift** = Support(X, Y) / Support(X)*Support(Y) : Y is three times more likely to be bought when X is bought than when it is bought independently.

> *python file*:  [association_rule_learning.py](association_rule_learning.py)

- Aim: The step-by-step codes to generate association rules are as follows: data preprocessing, preparing invoice-product matrix, extracting association rules, recommending products to users, and script.

- Method: *apriori* and *assocotion_rules* functions are used from *mlextend.frequent_patterns* library. 

- dataset: online_retail_II.xlsx [data link](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) 


*md file*: [association_rule_learning.md](association_rule_learning.md)



## PROJECT: ARMUT ARL RECOMMENDER SYSTEM

> *python file*:  [armut_arl.py](armut_arl.py)

- Aim: Create a product recommendation system with association rule learning method to recommended services to a user that having a certain service in the basket.

- Method: *apriori* and *assocotion_rules* functions are used from *mlextend.frequent_patterns* library. 

- dataset: armut.csv (Note: This dataset is not shared due to Miuul course confidentiality.)

*md file*: [armut_arl.md](armut_arl.md)



## PROJECT: ONLINE RETAIL ARL RECOMMENDER SYSTEM

> *python file*: [online_retail_arl.py](online_retail_arl.py)

- Aim: Recommend products to the user having product of 'product_id' in the basket using association rule learning method.

- Method: *apriori* and *assocotion_rules* functions are used from *mlextend.frequent_patterns* library. 

- dataset: online_retail_II.xlsx  [data link](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) 

*md file*: [online_retail_arl.md](online_retail_arl.md)



## CONTENT BASED RECOMMENDATION

Products with similar attributes to the products that user purchased will be recommended. 

Example: Content based recommender system will recommended you the movies with same genres or cast or similar topics to the movies that you already watched or gave high rating.

> *python file*: [content_based_recommendation.py](content_based_recommendation.py)

- Aim: Find the most similar movies to a given movie using overview text of the movies. 

- Method: The TF-IDF scores of the words from overview text are calculated. Then cosine similarities between movies are calculated. Then, the top 10 most similar movies to a specified movie based on highest cosine similarity scores.

- sklearn's *TfidfVectorizer* function is used to create tf-idf matrix. Matrix rows are the overviews and columns are the words.

- sklearn's *cosine_similarity* function is used to calculate cosine similarities between movies(overview texts).

- dataset: movies_metadata.csv  [data link](https://www.kaggle.com/rounakbanik/the-movies-dataset)

*md file*: [content_based_recommendation.md](content_based_recommendation.md)



## COLLABORATIVE FILTERING

User's past interactions are used to recommend new item. User-based and item-based are two main types of collaborative filtering.

## ITEM-BASED COLLABORATIVE FILTERING

The items having a reaction/preference from other users that are similar to what the user has liked in the past are recommended. The similarity of items is determined based on their ratings.

Example: If you liked a movie, recommender system recommends the similar movies to that movie based on the preferences (ratings) of other users who also liked it.

>`*python file*: [item_based_recommendation.py](item_based_recommendation.py)

- Aim: When users like a movie, recommend other movies with "similar liking patterns" to that movie.

- Method: The required data format user-movie-rating pivot table is created. The correlation between ratings of a selected movie and ratings of other movies are calculated. Then most similar movies are chosen based on correlation values.

- dataset: movie.csv, ratings_small.csv  [data link](https://grouplens.org/datasets/movielens/)

*md file*: [item_based_recommendation.md](item_based_recommendation.md)



## USER-BASED COLLABORATIVE FILTERING

Items that are purchased/used by the users with similar taste are recommended. The similarity of users is determined based on the ratings that are given.

Example: If user A and user B has similar ratings for the same movies, the movies that are liked by other user will be recommended to other user.

> *python file*: [user_based_recommendation.py](user_based_recommendation.py)

- Aim: Recommend movies to the users using using user-based collaborative filtering method.

- Method: The users that watched more than a specified percentage of movies with the selected user (to make recommendations) are chosen. The correlations for rating pattern between the selected user and other users are calculated. The weighted rating score is calculated for each movie-rating by multiplying the correlation and the movie's rating. Then, the average weighted rating is calculated for each movie. The movies with an average weighted rating greater than the specified threshold 'score' are recommended to the selected user.

- dataset: movie.csv, ratings_small.csv   [data link](https://grouplens.org/datasets/movielens/)

*md file*: [user_based_recommendation.md](user_based_recommendation.md)



## PROJECT: HYBRID RECOMMENDER SYSTEM

> *python file*: [hybrid_recommender.py](hybrid_recommender.py)

- Aim: Recommend 10 movies to the user using both user-based and item-based collaborative filtering methods.

- Method: Choose 5 movies from user-based recommendation and choose 5 movies from item-based recommendation.

- dataset: movie.csv, ratings.csv   [data link](https://grouplens.org/datasets/movielens/)

note: download ratings.csv from the link.

*md file*: [hybrid_recommender.md](hybrid_recommender.md)


## MODEL BASED MATRIX FACTORIZATION 

MF can be used to calculate the similarity in user’s ratings or interactions to provide recommendations. 

    user-item matrix : u x i

    user factor matrix : u x f

    item factor matrix : f x i 

    (u x i) = (u x f) * (f x i)

Empty rating values ​​in the user-item matrix are calculated by multiplying the user factors (user of interest row in the user factor matrix) and product factors (item of interest column in the item factor matrix).

Example: The rating of user 1 that would give to movie C will be predicted by using the rating values of other users that have similar rating behaviors with user 1 for the same movies. 

> *python file*: [matrix_factorization.py](matrix_factorization.py)

- Aim: All movies are not rated by all users so predict the missing rating values.

- Method: SVD model from scikit-suprise is used.

- dataset: movie.csv, ratings_small.csv   [data link](https://grouplens.org/datasets/movielens/)

*md file*: [matrix_factorization.md](matrix_factorization.md)