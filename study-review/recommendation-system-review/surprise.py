from surprise import SVD, Dataset, accuracy
from surprise.model_selection import train_test_split

data = Dataset.load_builtin("ml-100k")
trainset, testset = train_test_split(data, test_size=.25, random_state=0)

# 학습
algo = SVD(random_state=0)
algo.fit(trainset)

predictions = algo.test(testset)
print("prediction type:", type(predictions), "size:", len(predictions))
print("extract first 5")
print("predictions:", predictions[:5])

print([ (prd.uid, prd.iid, prd.est) for prd in predictions[:3] ])


uid = str(196)
iid = str(302)
prd = algo.predict(uid, iid)
print(prd)

accuracy.rmse(predictions)


import pandas as pd

ratings = pd.read_csv("/Users/jinjae/Code/Study/Python-Machine-Learning/ml-latest-small/ratings.csv")

ratings.to_csv("/Users/jinjae/Code/Study/Python-Machine-Learning/ml-latest-small/ratings_noh.csv", index=False, header=False)

from surprise import Reader

reader = Reader(line_format="user item rating timestamp", sep=',', rating_scale=(0.5, 5))
data = Dataset.load_from_file("/Users/jinjae/Code/Study/Python-Machine-Learning/ml-latest-small/ratings_noh.csv", reader=reader)

trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
