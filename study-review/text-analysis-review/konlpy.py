import pandas as pd

train_df = pd.read_csv("../../dataset/etc/ratings_train.txt", sep="\t")


import re

train_df = train_df.fillna(' ')
train_df["document"] = train_df["document"].apply(lambda x: re.sub(r"\d+", " ", x))

test_df = pd.read_csv("../../dataset/etc/ratings_test.txt", sep="\t")
test_df = test_df.fillna(' ')
test_df["document"] = test_df["document"].apply(lambda x: re.sub(r"\d+", " ", x))

train_df.drop("id", axis=1, inplace=True)
test_df.drop("id", axis=1, inplace=True)

from konlpy.tag import Okt

okt = Okt()


def okt_tokenizer(text):
    tokens_ko = okt.morphs(text)
    return tokens_ko


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf_vect = TfidfVectorizer(tokenizer=okt_tokenizer, ngram_range=(1, 2), min_df=3, max_df=0.9)
tfidf_vect.fit(train_df["document"])
tfidf_matrix_train = tfidf_vect.transform(train_df["document"])


from sklearn.linear_model import LogisticRegression

lg_clf = LogisticRegression(random_state=0, solver="liblinear")

params = {
    'C': [1, 3.5, 4.5, 5.5, 10]
}
grid_cv = GridSearchCV(lg_clf, param_grid=params, cv=3, scoring="accuracy", verbose=1)
grid_cv.fit(tfidf_matrix_train, train_df["label"])
print(grid_cv.best_params_, round(grid_cv.best_score_, 4))


from sklearn.metrics import accuracy_score

tfidf_matrix_test = tfidf_vect.transform(test_df["document"])

best_estimator = grid_cv.best_estimator_
preds = best_estimator.predict(tfidf_matrix_test)

print("Logistic Regression accuracy:", accuracy_score(test_df["label"], preds))
