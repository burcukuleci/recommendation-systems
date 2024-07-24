#############################
# Content Based Recommendation
#############################

#############################
# Make Recommendations using Overview's of the Movies
#############################

# 1. TF-IDF Matrix
# 2. Cosine Similarity Matrix
# 3. Make Recommendation According to Similarities
# 4. Script

# BUSINESS PROBLEM:
# Since user login behavior is low on the online movie platform, product recommendations cannot be developed according to user habits with collaborative filtering methods.
# It is known which movie the users watched only from the browsing history, so other movies with similar features can be recommended to the user.


#################################
# 1. TF-IDF Matrix
#################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("datasets/movies_metadata.csv", low_memory=False)
df.head()
df.shape

df["overview"].head()

# check if there is null values
df[df['overview'].isnull()]

# fill empty overviews with empty string
df['overview'] = df['overview'].fillna('')

# do not include stopwords
tfidf = TfidfVectorizer(stop_words="english")

# smaller feature numbers (words) can be used
tfidf = TfidfVectorizer(stop_words="english", max_features=10000)

# create TF-IDF matrix for 'overview' column values - sparse matrix -
tfidf_matrix = tfidf.fit_transform(df['overview'])
# tfidf_matrix rows are overviews.
# tfidf_matrix columns are the words.

tfidf_matrix.shape
# (45466, 75827) - row: movie overviews, columns: words

tfidf_matrix=tfidf_matrix[0:10000]   # running the script for small amount of movies

# get the words in the columns
tfidf.get_feature_names_out()

# convert sparse matrix to dense matrix
dense_tfidf_matrix=tfidf_matrix.toarray()



#################################
# 2. Cosine Similarity Matrix
#################################

# check the cosine similarities between movies using sparse matrix tfidf_matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)



#################################
# 3. Make Recommendation According to Similarities
#################################

# get the index numbers of movie titles.
indices = pd.Series(df.index, index=df['title'])

# check if there is any movie name occurs multiple times
indices.index.value_counts()

# there is duplication of some movie names. Only the recent movie will be kept among duplicated ones.
indices = indices[~indices.index.duplicated(keep='last')]

# Let's analysis the movie "Toy Story"
# get the movie index of "Toy Story":0
movie_index = indices["Toy Story"]

# cosine_sim[movie_index]: cosine similarities between "Toy Story" and other movies
# create dataframe for cosine similarities -- index: movie indices, column: similarity scores

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

# sort the similarity scores in decending order then select the indices of top 10 most similar movies to Toy Story.
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

# movie titles of top 10 most similar movies to Toy Story
df['title'].iloc[movie_indices]

# list of top 10 movies to recommend.
top10_movie_titles=df['title'].iloc[movie_indices].values.tolist()



#################################
# 4. Script
#################################


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    # matrix with tf-idf scores of words from overview texts.
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    # calculate cosine similarities between movies(overview texts).
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def content_based_recommender(title, cosine_sim, dataframe):
    # get the indices of movies -- index: title, values: index
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    # keep only the recent movie among duplicated ones.
    indices = indices[~indices.index.duplicated(keep='last')]
    # index of the movie title
    movie_index = indices[title]
    # calculate similarity score between 'title' and other movies
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # select top 10 most similar movies -- top 1 will be the movie itself
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

    # top 10 most similar movie titles with their indices
    return dataframe['title'].iloc[movie_indices]


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('Toy Story', cosine_sim, df)

