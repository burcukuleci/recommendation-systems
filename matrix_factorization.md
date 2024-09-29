# MODEL BASED MATRIX FACTORIZATION 

Empty rating values ​​in the user-item matrix are calculated by multiplying the user factors (user of interest row in the user factor matrix) and product factors (item of interest column in the item factor matrix).

Example: The rating of user 1 that would give to movie C will be predicted by using the rating values of other users that have similar rating behaviors with user 1 for the same movies. 

- Method: SVD model from scikit-suprise is used.

- dataset: movie.csv, ratings_small.csv   [data link](https://grouplens.org/datasets/movielens/)

1. [Prepare the Dataset](#1-prepare-the-dataset)
2. [Modelling](#2-modelling)
3. [Model Tuning](#3-model-tuning)
4. [Final Model and Prediction](#4-final-model-and-prediction)
5. [Predict the Non-existing Ratings](#5-predict-the-non-existing-ratings)


## 1. Prepare the Dataset

- Create user-movie pivot table 'user_movi_df' for only 4 movies with movie_ids of [1, 356, 4422, 541].

- Create required data 'data' format in order to use SVD model.

```python
reader = Reader(rating_scale=(1, 5))

# create SVD dataset.
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)
```

## 2. Modelling

Predict the rating for user(uid)=1.0 and movie(iid)=541.

- Split 'data' into trainset and testset.

- Train SVD model with default hyperparameter values.

```python
svd_model = SVD()
svd_model.fit(trainset)
```
- Predict the rating 'est'.

```python
svd_model.predict(uid=2.0, iid=541, verbose=True)
```

Output:

```
r_ui=actual rating    est=estimated rating
user: 2.0        item: 541        r_ui = None   est = 3.92   {'was_impossible': False}
```

## 3. Model Tuning

Perform hyperparameter tuning using GridSearchCV and choose the best model based on rmse metric.

```python
param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}

gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

# fit df to SVD Dataset object 'data'
gs.fit(data)
```

## 4. Final Model and Prediction

- Define the best SVD model 'svd_model_best' and train the model using full dataset.

```python
svd_model_best = SVD(**gs.best_params['rmse'])

# convert full 'data' to Trainset object in order to fit SVD model.
full_trainset = data.build_full_trainset()

# fit the model. 
svd_model_best.fit(full_trainset)
```

## 5. Predict the Non-existing Ratings

Create 'user_movie_df_filled' by filling empty rating values with the predicted rating values with SVD model.

- Predict ratings of movies in tuple list 'missing_ratings'. : predicted_ratings

```python
predicted_ratings = []

for (user_id, title) in missing_ratings:
    movie_id = sample_df[sample_df['title'] == title]['movieId'].iloc[0]
    prediction = svd_model_best.predict(uid=user_id, iid=movie_id)
    # list of tuples of (userid, title, predicted rating)
    predicted_ratings.append((user_id, title, prediction.est))
```

- Fill the empty ratings in 'user_movie_df' by ratings from 'predicted_ratings'. 

```python
for (user_id, title, rating) in predicted_ratings:
    user_movie_df_filled.at[user_id, title] = rating
```

