############################################
# User-Based Collaborative Filtering
#############################################

# 1: Preparing the Data Set
# 2: Determining the Movies Watched by the User to Make a Recommendation
# 3: Accessing Data and IDs of Other Users Watching the Same Movies
# 4: Identifying Users with the Most Similar Behavior to the User to Make a Recommendation
# 5: Calculating the Weighted Average Recommendation Score
# 6: Functionalization of Work

# BUSINESS PROBLEM: The online movie platform has made general suggestions by trying content-based and item-based suggestion systems based
#                   on similar liking structures for the movie, but it wants to make more customization based on the "similarity of users to users".



#############################################
# 1: Preparing the Data Set
#############################################

import pandas as pd

# all columns will be shown.
pd.set_option('display.max_columns', None)
# maximum width of the display in characters
pd.set_option('display.width', 500)
# not expand the dataframe to multiple lines if it exceeds the specified width.
pd.set_option('display.expand_frame_repr', False)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie.csv')
    rating = pd.read_csv('datasets/ratings_small.csv')
    df = movie.merge(rating, how="left", on="movieId")
    # number of ratings for each movie
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    # common movies that are rated more than 1000 times
    common_movies = df[~df["title"].isin(rare_movies)]
    # create pivot table
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


user_movie_df = create_user_movie_df()
user_movie_df.shape

# select a random user from the userids in the index from 'user_movie_df'.
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values[0])
# 196


#############################################
# 2: Determining the Movies Watched by the User to Make a Recommendation
#############################################

# get the ratings done by the random_user
random_user_df = user_movie_df[user_movie_df.index == random_user]

# select only the movies that the user watched/rated.
# bring columns (movie titles) that are not NaN - True values will be chosen with notna()
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# number of movies that watched by the user
len(movies_watched)

# get the rating for chosen movie from user_movie_df
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]



#############################################
# 3: Accessing Data and IDs of Other Users Watching the Same Movies
#############################################

# 'movies_watched': list of movies watched by the random_user / 191 movies /

# -- all users' ratings for the movies that are watched by 'random_user'
movies_watched_df = user_movie_df[movies_watched]   # index: title, column index: userId

# every movie is not watched by every user
# count the number of movies that watched by the users (who have watched at least one of the movies watched by the random_user)
# take transpose to move titles to index and userIds to columns then count ratings (count will be performed on columns)
user_movie_count = movies_watched_df.T.notnull().sum()
# make userId a column by resetting index
user_movie_count = user_movie_count.reset_index()
# name the columns
user_movie_count.columns = ["userId", "movie_count"]

# choose only the users having more than 15 "movie_count"
user_movie_count[user_movie_count["movie_count"] > 15].sort_values("movie_count", ascending=False)

# number of users that watched the same number of movies with random_user
user_movie_count[user_movie_count["movie_count"] == len(movies_watched)].count()

# -- Let's choose users that watched more than specified percentage of movies
# user_same_movies pandas series: values are the userId's from user_movie_count
users_same_movies = user_movie_count[user_movie_count["movie_count"] > int(len(movies_watched)*0.60)]["userId"]

# -- list of userIds that watched more than specified percentage of movies
userids_same_movies = users_same_movies.values.tolist()

# filter user_movie_count for userIds that watched more than specified percentage of movies
user_movie_count[user_movie_count['userId'].isin(userids_same_movies)]

# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100



#############################################
# 4: Identifying Users with the Most Similar Behavior to the User to Make a Recommendation
#############################################

### Step 1:  merge random user's ratings and other users'(with similarity percentage greater than 60%) ratings

# movies_watched_df[movies_watched_df.index.isin(users_same_movies)]: ratings of userId(index)'s that watched at least 60% of movies that watched by random_user
# random_user_df[movies_watched] : ratings of random_user for watched movies
# >> In other words, concat the ratings given to the movies by users with a certain similarity percentage and random_user.

# final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
#                      random_user_df[movies_watched]])

final_df = pd.concat([movies_watched_df.loc[userids_same_movies, :],
                      random_user_df[movies_watched]])

# index: userId
# columns: movie names


### Step 2: Calculate the correlation matrix

# To calculate the correlation between users, the columns must be userIDs because corr() calculates the correlation between columns.
# In order to do that take tranpose of final_df to switch between index and columns.
corr_matrix = final_df.T.corr()


### Step 3: Sort the values

# To be able to sort correlation between users the data format pandas series where there are two index of userId and the value is correlation.
sorted_corr_series = corr_matrix.stack().sort_values()


### Step 4: Drop duplicates

# drop duplicates due to user1-user2 = user2-user1
corr_series = sorted_corr_series.drop_duplicates()

# step 2,3,4 together >>>

# correlations of userId pairs >>>
corr_series = final_df.T.corr().stack().sort_values().drop_duplicates()

# convert series to dataframe
corr_df = pd.DataFrame(corr_series, columns=["corr"])

# name the indices
corr_df.index.names = ['user_id_1', 'user_id_2']

# make indices columns by reseting index
corr_df = corr_df.reset_index()


### Step 5: Find Top Users

# create new dataframe from corr_df where user_id_1 is the random_user's id and correlation values are larger than 0.5.
# print only user_i d_2 and corr values.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.50)][["user_id_2", "corr"]].reset_index(drop=True)

# sort corr values in descending order.
top_users = top_users.sort_values(by='corr', ascending=False)

# change the name of column "user_id_2" to "userId"
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)


# merge 'rating.csv' top_users to add movieId and rating information.
rating = pd.read_csv('datasets/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
# userId, corr, movieId, rating

# drop random_user from users.
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]



#############################################
# 5: Calculating the Weighted Average Recommendation Score
#############################################

# Question: How to choose top users? according to correlation value or rating value?
# consider both parameters: corr * rating = weighted rating
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# calculate mean weighted average for each movie
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
# index: movieId, column: weighted_rating

recommendation_df = recommendation_df.reset_index()

recommendation_df["weighted_rating"].max()  # 34.17

# select movies with weighted_rating greater than 3.5 then sort in descending order.
movies_to_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating",
                                                                                                   ascending=False)

movie = pd.read_csv('datasets/movie.csv')

# add movie titles
movies_to_recommend=movies_to_recommend.merge(movie[["movieId", "title"]])




#############################################
# # 6: Functionalization of Work
#############################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie.csv')
    rating = pd.read_csv('datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    # number of ratings for each movie
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    # common movies that are rated more than 1000 times
    common_movies = df[~df["title"].isin(rare_movies)]
    # create pivot table
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


user_movie_df = create_user_movie_df()

# bring other steps into function 'user_based_recommender'

def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    import pandas as pd
    # titles and ratings for random_user
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    # list of movies that are watched(rated) by random_user
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    # all users' ratings for movies watched by random_user
    movies_watched_df = user_movie_df[movies_watched]
    # number of watched movies by users: user_movie_count
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    # threshold for number of common movies
    perc = len(movies_watched) * ratio / 100
    # userId's with common movie percentage greater than ratio%
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
    # list of userIds that watched more than specified percentage of movies
    userids_same_movies = users_same_movies.values.tolist()
    final_df = pd.concat([movies_watched_df.loc[userids_same_movies, :],
                          random_user_df[movies_watched]])

    # correlation between users: corr_df
    corr_df = final_df.T.corr().stack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    # choose top users having correlation greater than cor_th with random_user: top_users
    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)
    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

    # add rating information to top_users: top_users_rating.
    rating = pd.read_csv('datasets/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

    # calculate weighted_rating= corr * rating
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    # calculate mean weighted_rating for each movie
    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    # return movieId, weighted_rating and title information for movies with mean weighted_rating greater than 'score'
    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values(
        "weighted_rating", ascending=False)
    movie = pd.read_csv('datasets/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])

# select a random_user
random_user = int(pd.Series(user_movie_df.index).sample(1).values[0])
recommended_movies=user_based_recommender(random_user, user_movie_df, cor_th=0.60, score=3.5)

# export titles of movies to be recommended
import pandas as pd
recommended_movies["title"].to_csv("recommended_movies.csv")