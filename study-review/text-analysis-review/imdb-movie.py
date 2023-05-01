import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

review_df = pd.read_csv("../../dataset/etc/labeledTrainData.tsv", header=0, sep="\t", quoting=3)
review_df.head()

review_df["review"] = review_df["review"].str.replace("<br />", " ")
review_df["review"] = review_df["review"].apply(lambda x : re.sub("[^a-zA-Z]", " ", x))

class_df = review_df["sentiment"]
feature_df = review_df.drop(["id", "sentiment"], axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(feature_df, class_df, test_size=0.3, random_state=156)

pipeline = Pipeline([
    # ("cnt_vect", CountVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("tfidf_vect", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("lr_clf", LogisticRegression(solver="liblinear", C=10))
])

pipeline.fit(X_train["review"], y_train)
prd = pipeline.predict(X_test["review"])
prd_probs = pipeline.predict_proba(X_test["review"])[:, 1]

print("accuracy: {0:.4f}, ROC-AUC score: {1:.4f}".format(accuracy_score(y_test, prd), roc_auc_score(y_test, prd_probs)))
