#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# We will use matrix-factorization based model SVD.
import pandas as pd
# conda install -c conda-forge scikit-surprise
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split

pd.set_option('display.max_columns', None)

# 1. Prepare the Dataset
# 2. Modelling
# 3. Model Tuning
# 4. Final Model and Prediction
# 5. Predict the Non-existing Ratings



#############################
# 1. Prepare the Dataset
#############################

# read csv files and merge.
movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/ratings_small.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

# choose only 4 movies.
movie_ids = [1, 356, 4422, 541]
movies = ["Toy Story (1995)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

# create sample_df from df for only selected 4 movies.
sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

# we only need userId, title and rating information from sample_df. Show them as pivot table.
user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape    # (452, 4) - 452 users, 4 movies

# total rating count.
total_rating_no = user_movie_df.size  # 452 users * 4 movies = 1808 ratings

# number of missing ratings.
missing_rating_no = user_movie_df.isnull().sum().sum()  # 1069 --> 59% of ratings are missing.

existing_rating_no = total_rating_no - missing_rating_no   # same with sample_df.shape

# --- create required data format in order to use SVD model.

# define rating scale range range.
reader = Reader(rating_scale=(1, 5))

# create SVD dataset.
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)



##############################
# 2. Modelling
##############################

# split data into trainset and testset.
trainset, testset = train_test_split(data, test_size=.25, random_state=42)

# SVD model with default hyperparameter values.
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)

# calculate accuracy score.
accuracy.rmse(predictions)   # 0.90

# predict the rating for user(uid)=1.0 and movie(iid)=541
svd_model.predict(uid=2.0, iid=541, verbose=True)

# Output >>>  r_ui=actual rating    est=estimated rating
# user: 2.0        item: 541        r_ui = None   est = 3.92   {'was_impossible': False}



##############################
# 3. Model Tuning
##############################

# 'n_epochs': iteration number, 'lr_all' : learning rate

# define hyperparameter value ranges. - 3x3=9 combinations.
param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}

# use GridSearchCV with 3 cv folds to train the model for each hyperparameter value combinations.: combination x cv=9x3=27 fits.
gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

# fit df to SVD Dataset object 'data'
gs.fit(data)

gs.best_score['rmse']    # 0.89

gs.best_params['rmse']



##############################
# 4. Final Model and Prediction
##############################

# use the best hyperparameter values to build a model.
svd_model_best = SVD(**gs.best_params['rmse'])

# hyperparameter tuning is done so full dataset can be used for model training.
# convert full 'data' to Trainset object in order to fit SVD model.
full_trainset = data.build_full_trainset()

# fit the model. - fit requires Trainset object. -
svd_model_best.fit(full_trainset)

# estimated rating for userId=2 and movieId=541
svd_model_best.predict(uid=2.0, iid=541, verbose=True)

# create Trainset object for full dataset.
# predictions will be done for al missing rating values in 'data'.
full_testset = full_trainset.build_testset()

# predictions for testset object. - test requires Testset objet -
predictions = svd_model_best.test(full_testset)

# print first 5 predictions
for prediction in predictions[:5]:
    print(prediction)

# convert predictions list to a dataframe - predictions for testset object -
results = pd.DataFrame(predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])

#        uid   iid  r_ui       est                    details
# 0      7.0     1   3.0  3.622625  {'was_impossible': False}
# 1      7.0   356   3.0  3.928325  {'was_impossible': False}

results.shape   # (739, 5) - row number equals to number of existing ratings -



##############################
# 5.Predict the Non-existing Ratings
##############################

### user-movie info for movies not rated by some users.
missing_ratings = user_movie_df.isnull().stack()
# user_Id, movie title tuples
missing_ratings = missing_ratings[missing_ratings].index.tolist()

# predict ratings
predicted_ratings = []

for (user_id, title) in missing_ratings:
    movie_id = sample_df[sample_df['title'] == title]['movieId'].iloc[0]
    prediction = svd_model_best.predict(uid=user_id, iid=movie_id)
    # list of tuples of (userid, title, predicted rating)
    predicted_ratings.append((user_id, title, prediction.est))

print(predicted_ratings)

# convert predicted ratings to a dataframe
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=['userId', 'title', 'predicted_rating'])

# create copy of user_movie_df to be filled.
user_movie_df_filled = user_movie_df.copy()

# add predicted ratings where rating is NaN
for (user_id, title, rating) in predicted_ratings:
    user_movie_df_filled.at[user_id, title] = rating

# estimated and real rating values for 4 selected movies >>
print(user_movie_df_filled)




