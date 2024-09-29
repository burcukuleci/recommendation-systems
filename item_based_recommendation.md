# ITEM-BASED COLLABORATIVE FILTERING

The items having a reaction/preference from other users that are similar to what the user has liked in the past are recommended.

If you liked a movie, recommender system recommends the similar movies to that movie based on the preferences (ratings) of other users who also liked it.

- dataset: movie.csv, ratings_small.csv  [link](https://grouplens.org/datasets/movielens/)

note: small rating dataset is used for better model performance and disk space.


## 1. Prepare Dataframe

- Read 'movie.csv' and 'ratings_small.csv' files then merge them into a dataframe.

```python
df = movie.merge(rating, how="left", on="movieId")
``` 

## 2. Create User-Movie Matrix

Filter movies that have rating counts smaller than a specified value.

- Create required data format as a pivot table 'user_movie_df': index=userId, column= movie title, value=rating

```python
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
``` 

## 3. Item-Based Movie Recommendations

- Find correlation coefficients between ratings 'movie_ratings' of selected 'movie_name' and other movies 'user_movie_df'. sort the calues in descending order and print top 10 most similar movies.

```python
user_movie_df.corrwith(movie_ratings).sort_values(ascending=False).head(10)
``` 

## 4. Script

- Create user-movie pivot table

```python
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie.csv')
    rating = pd.read_csv('datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    # number of ratings for each movie
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 50].index
    # common movies that are rated more than 50 times
    common_movies = df[~df["title"].isin(rare_movies)]
    # create pivot table in order to calculate correlations.
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()
``` 

- Define a funtion to get the list of recommend movies for specified 'movie_name'.

```python
def item_based_recommender(movie_name, user_movie_df):
    # pivot table for specified movie
    movie_ratings = user_movie_df[movie_name]
    # calculate correlation coefficients between the movie and other movies
    # select top 10 movies with most similar rating behavior
    top10_movies = user_movie_df.corrwith(movie_ratings).sort_values(ascending=False)[1:11]

    # return top 10 movie titles
    return top10_movies.index.to_list()

item_based_recommender("Matrix, The (1999)", user_movie_df)
``` 


