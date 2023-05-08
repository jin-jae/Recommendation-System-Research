import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


# 사용자가 평점을 부여한 영화에 한해서 예측 성능 평가를 함
def get_evaluation(prd, actual):
    # ignore nonzero
    prd = prd[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(prd, actual), r2_score(prd, actual)


# 아이템 기반의 협업 필터링에서 개인화된 예측 평점 도출 공식
def predict_rating(ratings_arr, item_sim_arr):
    ratings_prd = ratings_arr.dot(item_sim_arr) / np.sum(np.abs(item_sim_arr))
    return ratings_prd


def cos_similarity(v1, v2):
    print("v1.shape[0]", v1.shape[0])
    similarity = np.zeros((int(v1.shape[0]), int(v1.shape[0])))
    print("similarity:", similarity.shape)

    for i in range(similarity.shape[0]):
        similarity[i] = np.dot(v1[i], v2[i]) / (np.linalg.norm(v1[i]) * np.linalg.norm(v2[i]))

    return similarity


# def cosine_similarity(X, Y=None, dense_output=True):
#     X, Y = check_pairwise_arrays(X, Y)
#     X_normalized = normalize(X, copy=True)
#     if X is Y:
#         Y_normalized = X_normalized
#     else:
#         Y_normalized = normalize(Y, copy=True)
#     K = safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=dense_output)
#     return K


def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    # 예측 행렬 0으로 초기화
    prd = np.zeros(ratings_arr.shape)
    print("### predict_rating_topsim - ratings_arr ###\n", ratings_arr, "\n", ratings_arr.shape)
    print("### predict_rating_topsim - item_sim_arr ###\n", item_sim_arr, "\n", item_sim_arr.shape)

    # 사용자-아이템 평점 행렬 열 크기만큼 반복
    for col in range(ratings_arr.shape[1]):
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n-1:-1]]
        # 개인화된 평점 계산
        for row in range(ratings_arr.shape[0]):
            ratings_prd = predict_rating(ratings_arr[row, :][top_n_items].T, item_sim_arr[col, :][top_n_items])
            prd[row, col] = ratings_prd
    return prd


def get_unseen_movies(ratings_matrix, userId):
    user_rating = ratings_matrix.loc[userId, :]
    already_seen = user_rating[user_rating > 0].index.tolist()
    movies_list = ratings_matrix.columns.tolist()
    unseen_list = [movie for movie in movies_list if movie not in already_seen]
    return unseen_list


def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies


if __name__ == "__main__":
    # movies 는 movieId, title, genres 를 column 으로 가지고 있음
    movies = pd.read_csv("/Users/jinjae/Code/Study/Python-Machine-Learning/ml-latest-small/movies.csv")
    # ratings 는 userId, movieId, rating, timestamp 를 column 으로 가지고 있음
    ratings = pd.read_csv("/Users/jinjae/Code/Study/Python-Machine-Learning/ml-latest-small/ratings.csv")

    # timestamp 는 필요 없음. userId, movieId, rating 만 활용
    ratings = ratings[["userId", "movieId", "rating"]]

    # 영화 제목에 대한 유저들의 평가를 적어둔 행렬 반환을 위해 merge (JOIN) 작업 수행
    rating_movies = pd.merge(ratings, movies, on="movieId")

    # userId와 영화 제목을 pivot 해서 추천 예측을 위한 행렬 생성. NaN (null) 값은 0으로 채움.
    ratings_matrix = rating_movies.pivot_table("rating", index="userId", columns="title")
    ratings_matrix = ratings_matrix.fillna(0)
    print("ratings_matrix:\n", ratings_matrix.head())

    # 아이템 기반으로 추출하기 위해 transpose 적용 (적용 결과 각 행에는 title, 열에는 user의 평가가 들어가 있음)
    # userId    1   2   3...
    # title     ...
    # title1    ...
    # title2    ...
    # title3    ...
    ratings_matrix_T = ratings_matrix.transpose()
    print("ratings_matrix_T:\n", ratings_matrix_T.head())

    # 각 영화를 기준으로 유저의 평가 양상이 얼마나 유사한지 cosine similarity 에 넣어서 평가
    # (n, n)의 정사각 행렬 반환 (대각 행렬은 자기 자신에 대한 유사도, 1)
    item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)
    # cosine similarity 행렬에 제목을 붙여서 반환
    item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns, columns=ratings_matrix.columns)
    print(item_sim_df.shape)
    print(item_sim_df.head())

    plt.figure(figsize=(10, 8))
    sns.heatmap(item_sim_df, cmap='YlGnBu')
    plt.title('Item Similarity Heatmap')
    plt.xlabel('Movies')
    plt.ylabel('Movies')
    plt.show()


    # extract similar with The Godfather
    print(item_sim_df["Godfather, The (1972)"].sort_values(ascending=False)[1:6])
    print(item_sim_df["Inception (2010)"].sort_values(ascending=False)[1:6])

    ratings_prd = predict_rating_topsim(ratings_matrix.values, item_sim_df.values, n=20)
    print("##### ratings_prd #####\n", ratings_prd, "\n", ratings_prd.shape, "\n\n")
    mse, r2 = get_evaluation(ratings_prd, ratings_matrix.values)
    print("$$ Evaluation $$\n - MSE: {}\n - R2 Score: {}".format(mse, r2))
    ratings_prd_matrix = pd.DataFrame(data=ratings_prd, index=ratings_matrix.index, columns=ratings_matrix.columns)

    user_rating_id = ratings_matrix.loc[9, :]
    print("유저 9의 평가 내용:", user_rating_id[user_rating_id > 0].sort_values(ascending=False)[:10])

    unseen_list = get_unseen_movies(ratings_matrix, 9)
    recomm_movies = recomm_movie_by_userid(ratings_prd_matrix, 9, unseen_list, top_n=10)
    recomm_movies = pd.DataFrame(data=recomm_movies.values, index=recomm_movies.index, columns=['prd_score'])

    print("최종 추천 결과:", recomm_movies)