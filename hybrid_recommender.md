# PROJECT: HYBRID RECOMMENDER SYSTEM

Make a prediction using the item-based and user-based recommender methods for the user whose ID is given.

Consider 5 suggestions from the user-based model, 5 suggestions from the item-based model, and finally make 10 suggestions from 2 models.

- dataset: movie.csv, ratings.csv   [link](https://grouplens.org/datasets/movielens/)

note: 

download ratings.csv from the link. (20 million rows)


## 1. Recommend 5 movies by user-based recommendation

- Create user_movie_df pivot table using *create_user_movie_df()* function. 

- Recommended 5 movies using *user_based_recommender* function for userId=512.

```python
recommended_movies=user_based_recommender(512, user_movie_df, cor_th=0.60, score=3.5)

top5_movies_from_user_based=recommended_movies[:5]
```

## 2. Recommend 5 movies by item-based recommendation

Find the correlation between the movie with highest rating given by the user that watched recently and the other movies.

- Recommend top 5 (most similar) movies having highest correlation values.

```python
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False)

movies_from_item_based = item_based_recommender(movie_name, user_movie_df)

top5_movies_from_item_based=movies_from_item_based[1:6]
```

## 3. Merge the outputs of two recommendation models

- Create list of recommended movies.

```python
top5_movies_user_list=list(recommended_movies["title"][:5].values)
top5_movies_item_list=list(movies_from_item_based[1:6].index)

# add two lists
top10_movies_user_item_based_list=top5_movies_user_list + top5_movies_item_list
```

- Convert the list to a dataframe and saved as csv file.

```python
top10_movies_df=pd.DataFrame(top10_movies_user_item_based_list, columns=["movie_name"])

# extract top 10 movie names as csv file.
top10_movies_df.to_csv("top10_movies_hybrid_recommender.csv")
```