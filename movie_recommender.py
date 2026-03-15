import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset 
movies = pd.read_csv("dataset/movies_list.csv")
# select column
movies = movies[['title','overview']]
#remove blank/missing value
movies.dropna(inplace=True)
#convert text into vectors
vectorizer = CountVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(movies['overview'])

#Similarity matrix
similarity = cosine_similarity(matrix)

def recommend(movie_name):
  movie_index = movies[movies['title']==movie_name].index[0]
  distances = similarity[movie_index]
  movie_list = sorted(list(enumerate(distances)), reverse=True,key=lambda x:x[1])[1:6]
  for i in movie_list:
    print(movies.iloc[i[0]].title)

# call the function for the test
user_movie_name = input("Enter your favorite movie : ")

recommend(user_movie_name)