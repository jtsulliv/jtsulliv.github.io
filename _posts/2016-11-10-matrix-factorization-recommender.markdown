---
title:  "Matrix Factorization for Movie Recommendations in Python"
date:   2016-11-10
tags: [recommender systems]

header:
  image: "matrix_factorization_recommenders/waterfall_fingerlakes.jpg"
  caption: "Photo Credit: Ginny Lehman"

excerpt: "Movie Recommender, Matrix Factorization, Latent Factor Models"
---

In this post, I'll walk through a basic version of low-rank matrix factorization for recommendations and apply it to a dataset of 1 million movie ratings available from the [MovieLens](http://grouplens.org/datasets/movielens/) project. The MovieLens datasets were collected by GroupLens Research at the University of Minnesota.

[Previously](https://beckernick.github.io/music_recommender/), I used item-based collaborative filtering to make music recommendations from raw artist listen-count data. I had a decent amount of data, and ended up making some pretty good recommendations. Collaborative filtering methods that compute distance relationships between items or users are generally thought of as "neighborhood" methods, since they center on the idea of "nearness." That's how I made the artist recommendations -- finding the artists with the closest vectors. Unfortunately, there are two issues with taking this approach:

1. It doesn't scale particularly well to massive datasets.
2. There's a theoretical concern with raw data based approaches.

I talked about the scaling issue in the previous post, but not the conceptual issue. The key concern is that ratings matrices may be overfit and noisy representations of user tastes and preferences. When we use distance based "neighborhood" approaches on raw data, we match on sparse, low-level details that we assume represent the user's preference vectors instead of matching on the vectors themselves. It's a subtle difference, but it's important.

For example, if I've listened to ten Red Hot Chili Peppers songs and you've listened to ten different Red Hot Chili Peppers songs, the raw user action matrix wouldn't have any overlap. Mathematically, the dot product of our action vectors would be 0. We'd be in entirely separate neighborhoods, even though it seems pretty likely we share at least some underlying preferences.

Using item features (such as genre) could help fix this issue, but not entirely. Stealing an example from Joseph Konston (professor at Minnesota who has a [Coursera course](https://www.coursera.org/specializations/recommender-systems) on recommender systems), what if we both like songs with great storytelling, regardless of the genre? How do we resolve this? I need a method that can derive tastes and preference vectors from the raw data.

Low-Rank Matrix Factorization is that kind of method.

# Matrix Factorization via Singular Value Decomposition

Matrix factorization is the breaking down of one matrix into a product of multiple matrices. It's extremely well studied in mathematics, and it's highly useful. There are many different ways to factor matrices, but singular value decomposition is particularly useful for making recommendations.

So what is singular value decomposition (SVD)? At a high level, SVD is an algorithm that decomposes a matrix $$R$$ into the best lower rank (i.e. smaller/simpler) approximation of the original matrix $$R$$. Mathematically, it decomposes $$R$$ into two unitary matrices and a diagonal matrix:

$$\begin{equation}
R = U\Sigma V^{T}
\end{equation}$$

where $$R$$ is user ratings matrix, $$U$$ is the user "features" matrix, $$\Sigma$$ is the diagonal matrix of singular values (essentially weights), and $$V^{T}$$ is the movie "features" matrix. $$U$$ and $$V^{T}$$ are orthogonal, and represent different things. $$U$$ represents how much users "like" each feature and $$V^{T}$$ represents how relevant each feature is to each movie.

To get the lower rank approximation, we take these matrices and keep only the top $$k$$ features, which we think of as the $$k$$ most important underlying taste and preference vectors.


# Setting Up the Ratings Data

Okay, enough with the math. Let's get to the code.


```python
import pandas as pd
import numpy as np

ratings_list = [i.strip().split("::") for i in open('/users/nickbecker/Downloads/ml-1m/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('/users/nickbecker/Downloads/ml-1m/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('/users/nickbecker/Downloads/ml-1m/movies.dat', 'r').readlines()]

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)
```

I'll also take a look at the movies and ratings dataframes.


```python
movies_df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MovieID</th>
      <th>Title</th>
      <th>Genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children's|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UserID</th>
      <th>MovieID</th>
      <th>Rating</th>
      <th>Timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1193</td>
      <td>5</td>
      <td>978300760</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>661</td>
      <td>3</td>
      <td>978302109</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>914</td>
      <td>3</td>
      <td>978301968</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3408</td>
      <td>4</td>
      <td>978300275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2355</td>
      <td>5</td>
      <td>978824291</td>
    </tr>
  </tbody>
</table>
</div>



These look good, but I want the format of my ratings matrix to be one row per user and one column per movie. I'll `pivot` `ratings_df` to get that and call the new variable `R_df`.


```python
R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
R_df.head()
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>MovieID</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>3943</th>
      <th>3944</th>
      <th>3945</th>
      <th>3946</th>
      <th>3947</th>
      <th>3948</th>
      <th>3949</th>
      <th>3950</th>
      <th>3951</th>
      <th>3952</th>
    </tr>
    <tr>
      <th>UserID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3706 columns</p>
</div>



The last thing I need to do is de-mean the data (normalize by each users mean) and convert it from a dataframe to a numpy array.


```python
R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
```

All set. With my ratings matrix properly formatted and normalized, I'm ready to do the singular value decomposition

# Singular Value Decomposition

Scipy and Numpy both have functions to do the singular value decomposition. I'm going to use the Scipy function `svds` because it let's me choose how many latent factors I want to use to approximate the original ratings matrix (instead of having to truncate it after).


```python
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)
```

Done. The function returns exactly what I detailed earlier in this post, except that the $$\Sigma$$ returned is just the values instead of a diagonal matrix. This is useful, but since I'm going to leverage matrix multiplication to get predictions I'll convert it to the diagonal matrix form.


```python
sigma = np.diag(sigma)
```

# Making Predictions from the Decomposed Matrices

I now have everything I need to make movie ratings predictions for every user. I can do it all at once by following the math and matrix multiply $$U$$, $$\Sigma$$, and $$V^{T}$$ back to get the rank $$k=50$$ approximation of $$R$$.

I also need to add the user means back to get the predicted 5-star ratings.


```python
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
```

If I wanted to put this kind of system into production, I'd want to create a training and validation set and optimize the number of latent features ($$k$$) by minimizing the Root Mean Square Error. Intuitively, the Root Mean Square Error will continuously decrease on the training set as $$k$$ increases (because I'm approximating the original ratings matrix with a higher rank matrix). On the validation set, however, the error will eventually start increasing because the training set is an overfit representation of user tastes.

For movies, predictions from lower rank matrices with values of $$k$$ between roughly 20 and 100 have been found to be the best at generalizing to unseen data.

I could create a training and validation set and optimize $$k$$ by minimizing RMSE, but I'll leave that for another post. I just want to see some movie recommendations.

# Making Movie Recommendations
Finally, it's time. With the predictions matrix for every user, I can build a function to recommend movies for any user. All I need to do is return the movies with the highest predicted rating that the specified user hasn't already rated. Though I didn't actually use any explicit movie content features (such as genre or title), I'll merge in that information to get a more complete picture of the recommendations.

I'll also return the list of movies the user has already rated, for the sake of comparison.


```python
def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False)
                 )

    print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
    print 'Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations)
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'MovieID',
               right_on = 'MovieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations

already_rated, predictions = recommend_movies(preds_df, 837, movies_df, ratings_df, 10)
```

    User 837 has already rated 69 movies.
    Recommending the highest 10 predicted ratings movies not already rated.


So, how'd I do?


```python
already_rated.head(10)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>UserID</th>
      <th>MovieID</th>
      <th>Rating</th>
      <th>Timestamp</th>
      <th>Title</th>
      <th>Genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>837</td>
      <td>858</td>
      <td>5</td>
      <td>975360036</td>
      <td>Godfather, The (1972)</td>
      <td>Action|Crime|Drama</td>
    </tr>
    <tr>
      <th>35</th>
      <td>837</td>
      <td>1387</td>
      <td>5</td>
      <td>975360036</td>
      <td>Jaws (1975)</td>
      <td>Action|Horror</td>
    </tr>
    <tr>
      <th>65</th>
      <td>837</td>
      <td>2028</td>
      <td>5</td>
      <td>975360089</td>
      <td>Saving Private Ryan (1998)</td>
      <td>Action|Drama|War</td>
    </tr>
    <tr>
      <th>63</th>
      <td>837</td>
      <td>1221</td>
      <td>5</td>
      <td>975360036</td>
      <td>Godfather: Part II, The (1974)</td>
      <td>Action|Crime|Drama</td>
    </tr>
    <tr>
      <th>11</th>
      <td>837</td>
      <td>913</td>
      <td>5</td>
      <td>975359921</td>
      <td>Maltese Falcon, The (1941)</td>
      <td>Film-Noir|Mystery</td>
    </tr>
    <tr>
      <th>20</th>
      <td>837</td>
      <td>3417</td>
      <td>5</td>
      <td>975360893</td>
      <td>Crimson Pirate, The (1952)</td>
      <td>Adventure|Comedy|Sci-Fi</td>
    </tr>
    <tr>
      <th>34</th>
      <td>837</td>
      <td>2186</td>
      <td>4</td>
      <td>975359955</td>
      <td>Strangers on a Train (1951)</td>
      <td>Film-Noir|Thriller</td>
    </tr>
    <tr>
      <th>55</th>
      <td>837</td>
      <td>2791</td>
      <td>4</td>
      <td>975360893</td>
      <td>Airplane! (1980)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>31</th>
      <td>837</td>
      <td>1188</td>
      <td>4</td>
      <td>975360920</td>
      <td>Strictly Ballroom (1992)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>28</th>
      <td>837</td>
      <td>1304</td>
      <td>4</td>
      <td>975360058</td>
      <td>Butch Cassidy and the Sundance Kid (1969)</td>
      <td>Action|Comedy|Western</td>
    </tr>
  </tbody>
</table>
</div>




```python
predictions
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MovieID</th>
      <th>Title</th>
      <th>Genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>516</th>
      <td>527</td>
      <td>Schindler's List (1993)</td>
      <td>Drama|War</td>
    </tr>
    <tr>
      <th>1848</th>
      <td>1953</td>
      <td>French Connection, The (1971)</td>
      <td>Action|Crime|Drama|Thriller</td>
    </tr>
    <tr>
      <th>596</th>
      <td>608</td>
      <td>Fargo (1996)</td>
      <td>Crime|Drama|Thriller</td>
    </tr>
    <tr>
      <th>1235</th>
      <td>1284</td>
      <td>Big Sleep, The (1946)</td>
      <td>Film-Noir|Mystery</td>
    </tr>
    <tr>
      <th>2085</th>
      <td>2194</td>
      <td>Untouchables, The (1987)</td>
      <td>Action|Crime|Drama</td>
    </tr>
    <tr>
      <th>1188</th>
      <td>1230</td>
      <td>Annie Hall (1977)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>1198</th>
      <td>1242</td>
      <td>Glory (1989)</td>
      <td>Action|Drama|War</td>
    </tr>
    <tr>
      <th>897</th>
      <td>922</td>
      <td>Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)</td>
      <td>Film-Noir</td>
    </tr>
    <tr>
      <th>1849</th>
      <td>1954</td>
      <td>Rocky (1976)</td>
      <td>Action|Drama</td>
    </tr>
    <tr>
      <th>581</th>
      <td>593</td>
      <td>Silence of the Lambs, The (1991)</td>
      <td>Drama|Thriller</td>
    </tr>
  </tbody>
</table>
</div>



Pretty cool! These look like pretty good recommendations. It's also good to see that, though I didn't actually use the genre of the movie as a feature, the truncated matrix factorization features "picked up" on the underlying tastes and preferences of the user. I've recommended some film-noirs, crime, drama, and war movies - all of which were genres of some of this user's top rated movies.

# Conclusion

We've seen that we can make good recommendations with raw data based collaborative filtering methods (neighborhood models) and latent features based matrix factorization methods (factorization models).

Low-dimensional matrix recommenders try to capture the underlying features driving the raw data (which we understand as tastes and preferences). From a theoretical perspective, if we want to make recommendations based on people's tastes, this seems like the better approach. This technique also scales **significantly** better to larger datasets, since we can actually approximate the SVD with gradient descent.

However, we still likely lose some meaningful signals by using a lower-rank matrix. And though these factorization based techniques work extremely well, there's research being done on new methods. These efforts have resulted in various types probabilistic matrix factorization (which works and scales even better) and many other approaches.

One particularly cool and effective strategy is to combine factorization and neighborhood methods into one [framework](http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf). This research field is extremely active, and I highly recommend Joseph Konstan's Coursera course, [Introduction to Recommender Systems](https://www.coursera.org/specializations/recommender-systems), for anyone looking to get a broad overview of the field.

***
For those interested, the Jupyter Notebook with all the code can be found in the [Github repository](https://github.com/beckernick/matrix_factorization_recommenders) for this post.
