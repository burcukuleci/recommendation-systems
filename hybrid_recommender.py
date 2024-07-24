
#############################################
# PROJECT: Hybrid Recommender System
#############################################

# Make a prediction using the item-based and user-based recommender methods for the user whose ID is given.
# Consider 5 suggestions from the user-based model, 5 suggestions from the item-based model, and finally make 10 suggestions from 2 models.

#############################################
# 1: Recommend 5 movies by user-based recommendation
#############################################

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 300)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie.csv')
    rating = pd.read_csv('datasets/ratings.csv')
    df = movie.merge(rating, how="left", on="movieId")
    # number of ratings for each movie
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 100].index
    # common movies that are rated more than 150 times
    common_movies = df[~df["title"].isin(rare_movies)]
    # create pivot table
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


user_movie_df = create_user_movie_df()
user_movie_df.notnull().sum().sum()

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
    rating = pd.read_csv('datasets/ratings.csv')
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
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state= 42).values[0])
recommended_movies=user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4)

# recommended movies for userId=512
recommended_movies=user_based_recommender(512, user_movie_df, cor_th=0.60, score=3.5)

# TOP 5 movies with user-based recommendation. for userId=512
top5_movies_from_user_based=recommended_movies[:5]


#############################################
# 2. Recommend 5 movies by item-based recommendation
#############################################

# Make an item-based recommendation based on the last and highest rated movie the user watched.

user = 512

## 2.1: read movie and rating csv files.
movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/ratings.csv')

rating[rating["userId"] == user]["rating"].max()
# 4.5 is the highest rating.


## 2.2: Find movieId of a movie with rated as highest(4.5) and watched most recently by the user.
# NOTE: that movie may not be in user_movie_df because it may be chosen as rare movie.

rating[(rating["userId"] == user) & (rating["rating"] == 4.5)].sort_values(by="timestamp", ascending=False)
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 4.5)].sort_values(by="timestamp", ascending=False)["movieId"][:1].values[0]



## 2.3: Filter the user_movie_df dataframe created in the user based recommendation section according to the selected movie_id.

movie_name=movie[movie["movieId"] == movie_id]["title"].values[0]

# movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]

# ratings for 'movie_name'
movie_df = user_movie_df[[movie_name]]
movie_df.notnull().sum()  


## 2.4: Find correlation between chosen movie (movie_name) and other movies. : between movies from user_movie_df and movie_name.
# user corrwith() function to calculate correlation between columns not column values.
# sort correleation in descending order and return top 5 movies. Note: first movie will be itself with correlation of 1.0.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)     # title and corr

# create function for correlation analysis.
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False)

## 2.5: choose top 5 movies that have correlation with movie_name (except the movie itself).

# find correlations using the function.
movies_from_item_based = item_based_recommender(movie_name, user_movie_df)

# TOP 5 movies with item-based recommendation.
top5_movies_from_item_based=movies_from_item_based[1:6]



#############################################
# 3. Merge the outputs of two recommendation models
#############################################

top5_movies_user_list=list(recommended_movies["title"][:5].values)
top5_movies_item_list=list(movies_from_item_based[1:6].index)

# add two lists
top10_movies_user_item_based_list=top5_movies_user_list + top5_movies_item_list

# convert list to a dataframe
top10_movies_df=pd.DataFrame(top10_movies_user_item_based_list, columns=["movie_name"])

# saved top 10 movie names as csv file.
top10_movies_df.to_csv("top10_movies_hybrid_recommender.csv")