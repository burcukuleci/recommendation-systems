###########################################
# Item-Based Collaborative Filtering
###########################################

# dataset: https://grouplens.org/datasets/movielens/

# 1. Prepare Dataframe
# 2. Create User Movie Df
# 3. Item-Based Movie Recommendations
# 4. Script

### Business Problem : When users like a movie, recommend other movies with "similar liking patterns" to that movie.



######################################
# 1. Prepare Dataframe
######################################

import pandas as pd
pd.set_option('display.max_columns', 500)

movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/ratings_small.csv')

# merge movie and rating on common column of 'movieId'
df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape

# number of movies -- 27262
df["title"].nunique()



######################################
# 2. Create User-Movie Df
######################################

# number of ratings for each movie
comment_counts = pd.DataFrame(df["title"].value_counts())

comment_counts.describe()
# mean rating count for a movie is maximum 341. (small dataset is used.)

# determine rare movies having less than 50 ratings
rare_movies = comment_counts[comment_counts["count"] <= 50].index

# determine common movies as movies that are not in rare_movies
common_movies = df[~df["title"].isin(rare_movies)]

common_movies["title"].nunique()   # 444 movies

# create pivot table --- index=userId, column= movie title, value=rating
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.isnull().sum()
# there are missing rating values due to movies that are not rated by some users.



######################################
# 3. Item-Based Movie Recommendations
######################################

movie_name = "Matrix, The (1999)"

# userId-rating for a 'movie_name'
movie_ratings = user_movie_df[movie_name]

# find correlation coefficients between ratings 'movie_ratings' of selected 'movie_name' and other movies 'user_movie_df'.
user_movie_df.corrwith(movie_ratings).sort_values(ascending=False).head(10)

# select a random movie from user_movies_df
# use values[0] to extract movie name
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

# movie ratings with userId for randomly selected movie
movie_ratings = user_movie_df[movie_name]

# find correlation between the movie and other movies
# then sort ratings in descending order and chose top 10 movies with most similar rating behavior
user_movie_df.corrwith(movie_ratings).sort_values(ascending=False).head(10)


# find movies that contains a specified keyword
def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]



######################################
# 4. Script
######################################

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


def item_based_recommender(movie_name, user_movie_df):
    # pivot table for specified movie
    movie_ratings = user_movie_df[movie_name]
    # calculate correlation coefficients between the movie and other movies
    # select top 10 movies with most similar rating behavior
    top10_movies = user_movie_df.corrwith(movie_ratings).sort_values(ascending=False)[1:11]

    # return top 10 movie titles
    return top10_movies.index.to_list()

# create user-movie pivot table
user_movie_df = create_user_movie_df()

# recommend movies for specified 'movie_name'
item_based_recommender("Matrix, The (1999)", user_movie_df)

# recommend 10 movies for a randomly selected movie
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
item_based_recommender(movie_name, user_movie_df)






