import pandas as pd
from surprise.model_selection import cross_validate
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

ratings = pd.read_csv("/Users/jinjae/Code/Study/Python-Machine-Learning/ml-latest-small/ratings.csv")
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

algo = SVD(random_state=0)
cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

from surprise.model_selection import GridSearchCV

param_grid = {"n_epochs": [20, 40, 60],
              "n_factors": [50, 100, 200]}

gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)
gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])

from surprise.dataset import DatasetAutoFolds

reader = Reader(line_format="user item rating timestamp", sep=',', rating_scale=(0.5, 5))
data_folds = DatasetAutoFolds(ratings_file="/Users/jinjae/Code/Study/Python-Machine-Learning/ml-latest-small/ratings_noh.csv", reader=reader)

# make all data to train set
trainset = data_folds.build_full_trainset()

algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)

movies = pd.read_csv("/Users/jinjae/Code/Study/Python-Machine-Learning/ml-latest-small/movies.csv")

movieIds = ratings[ratings["userId"] == 9]["movieId"]

if movieIds[movieIds == 42].count() == 0:
    print("movie 42: no review from user 9")

print(movies[movies["movieId"] == 42])

uid = str(9)
iid = str(42)

prd = algo.predict(uid, iid, verbose=True)


def get_unseen_surprise(ratings, movies, userId):
    seen_movies = ratings[ratings["userId"] == userId]["movieId"].tolist()
    total_movies = movies["movieId"].tolist()
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]
    print("graded movies:", len(seen_movies), "to recommends:", len(unseen_movies),
          "all movies:", len(total_movies))

    return unseen_movies


unseen_movies = get_unseen_surprise(ratings, movies, 9)


def recomm_movie_by_surprise(algo, userId, unseen_movies, top_n=10):
    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]

    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions= predictions[:top_n]

    top_movie_ids = [ int(pred.iid) for pred in top_predictions]
    top_movie_rating = [ pred.est for pred in top_predictions]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)]['title']
    top_movie_preds = [ (id, title, rating) for id, title, rating in zip(top_movie_ids, top_movie_titles, top_movie_rating)]

    return top_movie_preds

unseen_movies = get_unseen_surprise(ratings, movies, 9)
top_movie_preds = recomm_movie_by_surprise(algo, 9, unseen_movies, top_n=10)
print('##### Top-10 추천 영화 리스트 #####')

for top_movie in top_movie_preds:
    print(top_movie[1], ":", top_movie[2])
