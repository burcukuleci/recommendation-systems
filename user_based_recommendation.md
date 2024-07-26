# USER-BASED COLLABORATIVE FILTERING

Recommend movies to the users using using user-based collaborative filtering method.

- Method: The users that watched more than a specified percentage of movies with the selected user (to make recommendations) are chosen. The correlations for rating pattern between the selected user and other users are calculated. The weighted rating score is calculated for each movie-rating by multiplying the correlation and the movie's rating. Then, the average weighted rating is calculated for each movie. The movies with an average weighted rating greater than the specified threshold 'score' are recommended to the selected user.

- Dataset: movie.csv, ratings_small.csv   [link](https://grouplens.org/datasets/movielens/)

**note:** small rating dataset is used for better model performance and disk space. Download 'rating.csv' from the link and use instead of ratings_small.csv to obtain more accurate and meaningfull results.

1. [Preparing the Data Set](#1-preparing-the-data-set)
2. [Determining the Movies Watched by the User to Make a Recommendation](#2-determining-the-movies-watched-by-the-user-to-make-a-recommendation)
3. [Accessing Data and IDs of Other Users Watching the Same Movies](#3-accessing-data-and-ids-of-other-users-watching-the-same-movies)
4. [Identifying Users with the Most Similar Behavior to the User to Make a Recommendation](#4-identifying-users-with-the-most-similar-behavior-to-the-user-to-make-a-recommendation)
5. [Calculating the Weighted Average Recommendation Score](#5-calculating-the-weighted-average-recommendation-score)
6. [Functionalization of Work](#6-functionalization-of-work)


## 1. Preparing the Data Set

- Read 'movie.csv' and 'ratings_small.csv' files then merge them into a dataframe.

- Create required data format as a pivot table 'user_movie_df': index=userId, column= movie title, value=rating

```python
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
``` 

- select a user randomly from index of user_movie_df: random_user

```python
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values[0])
``` 

## 2. Determining the Movies Watched by the User to Make a Recommendation

- get the ratings given by the random_user

```python
random_user_df = user_movie_df[user_movie_df.index == random_user]
``` 

- list of movies watched by the random_user.

```python
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
``` 

## 3. Accessing Data and IDs of Other Users Watching the Same Movies

- user-movie dataframe with all users' ratings for the movies that are watched by 'random_user'.

```python
movies_watched_df = user_movie_df[movies_watched]
``` 

- count the number of movies that watched by the users: user_movie_count

- arrange 'user_movie_count' such that columns are userId an movie_count.

```python
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
``` 

- get the userIds that watched more than specified percentage of movies: userids_same_movies

```python
users_same_movies = user_movie_count[user_movie_count["movie_count"] > int(len(movies_watched)*0.60)]["userId"]
userids_same_movies = users_same_movies.values.tolist()
``` 

## 4. Identifying Users with the Most Similar Behavior to the User to Make a Recommendation

Compare similarities for rating pattern between the random_user and other users. (watched more than specified percentage of movies)

- final_df: concat (rows) ratings of the 'random_user' and 'users_same_movies'

```python
final_df = pd.concat([movies_watched_df.loc[userids_same_movies, :],
                      random_user_df[movies_watched]])
``` 

- Calculate correlation coefficients between users' ratings. : corr_df

Correlation is calculated between column values with corr() function so transpose of final_df is used.

```python
corr_series = final_df.T.corr().stack().sort_values().drop_duplicates()

# convert series to dataframe
corr_df = pd.DataFrame(corr_series, columns=["corr"])
# name the indices
corr_df.index.names = ['user_id_1', 'user_id_2']
# make indices columns by reseting index
corr_df = corr_df.reset_index()
``` 

        user_id_1  user_id_2      corr
    0        15.0       15.0  1.000000
    1       654.0      509.0  0.614231
    2       452.0      654.0  0.601629
    3       196.0      654.0  0.569058
    4       196.0      654.0  0.569058

- Find top users having correlation greater than 0.5 with the random_user.

```python
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.50)][["user_id_2", "corr"]].reset_index(drop=True)
# sort corr values in descending order.
top_users = top_users.sort_values(by='corr', ascending=False)
# change the name of column "user_id_2" to "userId"
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
``` 

top_users:

       userId      corr
    0   311.0  0.834428
    1   654.0  0.569058
    2   463.0  0.528302

- Bring the ratings of the top users together as a dataframe 'top_users_ratings'.

top_users_ratings:

          userId      corr  movieId  rating
    0      311.0  0.834428        1     3.0
    1      311.0  0.834428        6     4.0
    2      311.0  0.834428        7     3.0

## 5. Calculating the Weighted Average Recommendation Score

Until this step the top users are determined according to correlation value. But the effect of the rating values is neglected.

- Determine the top users(most similar) based on a score "weighted_rating" using both correlation values and ratings.

```python
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
``` 
- Calculate average weighted rating for each movie and choose the movies to recommend according to *average weighted rating*.

```python
movies_to_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)
``` 
- Final dataframe:

```python
movies_to_recommend=movies_to_recommend.merge(movie[["movieId", "title"]])
```
movies_to_recommend:

        movieId  weighted_rating                                              title
    0       529         4.172140                 Searching for Bobby Fischer (1993)
    1      8665         4.172140                       Bourne Supremacy, The (2004)
    2      1293         4.172140                                      Gandhi (1982)

# 6. Functionalization of Work

Define functions.

```python
user_movie_df = create_user_movie_df()
```

```python
recommended_movies=user_based_recommender(random_user, user_movie_df, cor_th=0.60, score=3.5)
```
