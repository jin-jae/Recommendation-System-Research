from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


news_data = fetch_20newsgroups(subset="all", random_state=156)

train_news = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), random_state=156)
X_train = train_news.data
y_train = train_news.target

test_news = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"), random_state=156)
X_test = test_news.data
y_test = test_news.target

pipeline = Pipeline([
    ("tfidf_vect", TfidfVectorizer(stop_words="english")),
    ("lr_clf", LogisticRegression())
])

params = {
    "tfidf_vect__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf_vect__max_df": [100, 300, 700],
    "lr_clf__C": [1, 5, 10]
}

grid_cv_pipe = GridSearchCV(pipeline, param_grid=params, cv=3, scoring="accuracy", verbose=1)
grid_cv_pipe.fit(X_train, y_train)
print(grid_cv_pipe.best_params_, grid_cv_pipe.best_score_)

prd = grid_cv_pipe.predict(X_test)
print("accuracy: {0:.3f}".format(accuracy_score(y_test, prd)))
