import numpy as np
import pandas as pd
import streamlit as st
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv('https://raw.githubusercontent.com/nextonnetflix/nextonnetflix/main/Training_Dataset.csv')

import nltk; nltk.download('punkt')

import nltk; nltk.download('stopwords')

desc = movies_data['description']

desclist = desc.values.tolist()
keylist = []
print(desclist)

!pip install rake-nltk

from rake_nltk import Rake

rake = Rake()
for i in desclist:
  rake.extract_keywords_from_text(i)
  a = rake.get_ranked_phrases()
  keylist.append(a)

combined_features = movies_data['type']+' '+movies_data['genre']+' '+movies_data['cast']+' '+movies_data['creators']

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)

close_match = find_close_match[0]
print(close_match)

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)

len(similarity_score)

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1

st.title('Movie Recommendation System') 

selected_movie_name=st.selectbox(
'Watch your favorite movie and enjoy your day',
movies['title'].values)

if st.button('Recommend'):
    recommendations=recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)



