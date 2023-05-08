import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def get_evaluation(R, P, Q, non_zeros):
    error = 0
    full_pred_matrix = np.dot(P, Q.T)

    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]

    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]

    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    r2 = r2_score(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)

    return rmse, r2


def matrix_factorization(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape
    np.random.seed(1)
    P = np.random.normal(scale=1./K, size=(num_users, K))
    Q = np.random.normal(scale=1./K, size=(num_items, K))

    non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i,j] > 0]

    for step in range(steps):
        for i, j, r in non_zeros:
            eij = r - np.dot(P[i, :], Q[j, :].T)
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse, r2 = get_evaluation(R, P, Q, non_zeros)
        if (step % 25) == 0:
            print("# iteration step:", step, "rmse:", rmse, "r2:", r2)

    return P, Q


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
    movies = pd.read_csv('/Users/jinjae/Code/Study/Python-Machine-Learning/ml-latest-small/movies.csv')
    ratings = pd.read_csv('/Users/jinjae/Code/Study/Python-Machine-Learning/ml-latest-small/ratings.csv')

    ratings = ratings[['userId', 'movieId', 'rating']]
    # ratings_matrix = ratings.pivot_table('rating', index='userId', columns='movieId')
    rating_movies = pd.merge(ratings, movies, on='movieId')
    ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title')
    print(ratings_matrix.head(), ratings_matrix.shape)

    # matrix factorization
    P, Q = matrix_factorization(ratings_matrix.values, K=50, steps=200, learning_rate=0.01, r_lambda=0.01)
    prd_matrix = np.dot(P, Q.T)

    ratings_prd_matrix = pd.DataFrame(data=prd_matrix, index=ratings_matrix.index, columns=ratings_matrix.columns)
    print(ratings_prd_matrix.head())

    unseen_list = get_unseen_movies(ratings_matrix, 9)
    recomm_movies = recomm_movie_by_userid(ratings_prd_matrix, 9, unseen_list, top_n=10)
    recomm_movies = pd.DataFrame(data=recomm_movies.values, index=recomm_movies.index, columns=["prd_score"])

    print(recomm_movies)
