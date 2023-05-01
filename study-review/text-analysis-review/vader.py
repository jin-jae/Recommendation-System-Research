from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.metrics import recall_score
import numpy as np
import nltk
import pandas as pd

nltk.download("all")

review_df = pd.read_csv("../../dataset/etc/labeledTrainData.tsv", header=0, sep="\t", quoting=3)

senti_analyzer = SentimentIntensityAnalyzer()
senti_scores = senti_analyzer.polarity_scores(review_df["review"][0])
print(senti_scores)


def vader_polarity(review, threshold=0.1):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)

    agg_score = scores["compound"]
    final_sentiment = 1 if agg_score >= threshold else 0
    return final_sentiment


review_df["vader_preds"] = review_df["review"].apply(lambda x: vader_polarity(x, 0.1))
y_target = review_df["sentiment"].values
vader_preds = review_df["vader_preds"].values

print(confusion_matrix(y_target, vader_preds))
print("accuracy:", np.round(accuracy_score(y_target, vader_preds), 4))
print("precision:", np.round(precision_score(y_target, vader_preds), 4))
print("recall:", np.round(recall_score(y_target, vader_preds), 4))
