# CONTENT BASED RECOMMENDATION

Make recommendations using overview's of the movies.

**Method:**

If word count is used, the frequency of the word in one movie description cannot be directly compared with the frequency values ​​of the words in the other movie description.

**Step1**: Calculate word counts for each description.

**Step2**: Calculate the TF(Term Frequency) value to standardize each movie within itself.
 
       Freq(word) / N , N: total words in the movie overview

**Step3**: Calculate IDF (Inverse Document Frequency) value. – How many of the movie descriptions in the corpus contain the relevant word?

**Step4**: Calculate TF-IDF score. TF-IDF = TF*IDF

**Step5**: L2 Normalization. TF-IDF/square root of the sum of squares of TF-IDF values

All of this steps are done within TfidfVectorizer() function.

- dataset: movies_metadata.csv  [link](https://www.kaggle.com/rounakbanik/the-movies-dataset)

1. [TF-IDF Matrix](#1-tf-idf-matrix)
2. [Cosine Similarity Matrix](#2-cosine-similarity-matrix)
3. [Make Recommendation According to Similarities](#3-make-recommendation-according-to-similarities)
4. [Script](#4-script)


## 1. TF-IDF Matrix

- Define tfidf. TfidfVectorizer is used to calculate TF-IDF scores of words.

```python
tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
```

- Create TF-IDF matrix for 'overview' column values - sparse matrix -

```python
tfidf_matrix = tfidf.fit_transform(df['overview'])
```

tfidf_matrix rows are overviews.

tfidf_matrix columns are the words.

## 2. Cosine Similarity Matrix

- Check the cosine similarities between movies(overviews) using sparse matrix tfidf_matrix.

```python
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

## 3. Make Recommendation According to Similarities

- Get the movie index of "Toy Story".

```python
movie_index = indices["Toy Story"]
```

- Create dataframe for cosine similarities. -- index: movie indices, column: similarity scores

cosine_sim[movie_index]: cosine similarities between "Toy Story" and other movies

```python
similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
```

- Sort the similarity scores in decending order then select the indices of top 10 most similar movies to Toy Story.

```python
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
```

- list of top 10 movies to recommend.

```python
top10_movie_titles=df['title'].iloc[movie_indices].values.tolist()
``` 

## 4. Script

Define functions for steps.

- define function to calculate cosine similarity score between movies.

```python
def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    # matrix with tf-idf scores of words from overview texts.
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    # calculate cosine similarities between movies(overview texts).
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)
```

- define function to determine 10 most similar movies based on similarity score.

```python
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

content_based_recommender('Toy Story', cosine_sim, df)
```

