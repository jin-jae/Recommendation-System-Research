import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

movies = pd.read_csv("../../dataset/etc/tmdb_5000_movies.csv")
print("movies.shape:", movies.shape)

movies_df = movies[["id", "title", "genres", "vote_average", "vote_count", "popularity", "keywords", "overview"]]

pd.set_option("max_colwidth", 100)
print(movies_df[["genres", "keywords"]][:1])


# string to dict
from ast import literal_eval
movies_df["genres"] = movies_df["genres"].apply(literal_eval)
movies_df["keywords"] = movies_df["keywords"].apply(literal_eval)

movies_df["genres"] = movies_df["genres"].apply(lambda x: [y["name"] for y in x])
movies_df["keywords"] = movies_df["keywords"].apply(lambda x: [y["name"] for y in x])
print(movies_df[["genres", "keywords"]][:1])

# 장르 콘텐츠 유사도 측정
from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer 적용을 위해 word 단위 구분 문자열로 반환
movies_df["genres_literal"] = movies_df["genres"].apply(lambda x: (' ').join(x))
count_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
genre_mat = count_vect.fit_transform(movies_df["genres_literal"])
print(genre_mat.shape)

from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)
print(genre_sim.shape)
print(genre_sim[:5])

# 유사도 값이 높은 순서대로 인덱스 값 추출
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
print("########## genre_sim_sorted_ind:\n", genre_sim_sorted_ind)


def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    title_movie = df[df["title"] == title_name]

    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]

    print("########## similar_indexes:\n", similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)

    return df.iloc[similar_indexes]


similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, "The Godfather", 10)
print(similar_movies[["title", "vote_average"]])

# 왜곡된 데이터 확인
print(movies_df[["title", "vote_average", "vote_count"]].sort_values("vote_average", ascending=False)[:10])

C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.6)
print("C:", round(C, 3), "m:", round(m, 3))

# 전체 투표 횟수 중 상위 60% 횟수 기준 추출
percentile = 0.6
m = movies_df["vote_count"].quantile(percentile)
C = movies["vote_average"].mean()


def weighted_vote_average(record):
    v = record["vote_count"]
    R = record["vote_average"]

    return ((v / (v + m)) * R) + ((m / (m + v)) * C)


movies_df["weighted_vote"] = movies.apply(weighted_vote_average, axis=1)

print(movies_df[["title", "vote_average", "weighted_vote", "vote_count"]].sort_values("weighted_vote", ascending=False)[:10])


similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, 'The Godfather',10)
print(similar_movies[['title', 'vote_average', 'weighted_vote']])
