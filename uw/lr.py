import pickle
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

root_path = 'gdrive/My Drive/Colab Notebooks/'

with open(root_path+'train_features.pkl', 'rb') as f:
    train_features = pickle.load(f)
with open(root_path+'train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f).values
with open(root_path+'test_features.pkl', 'rb') as f:
    test_features = pickle.load(f)
with open(root_path+'test_tplabels.pkl', 'rb') as f:
    test_labels = pickle.load(f).values

train_labels = np.squeeze(np.eye(2)[train_labels-1])
test_labels = np.squeeze(np.eye(2)[test_labels-1])

train_features = train_features.reshape(-1, train_features.shape[-1])
train_labels = train_labels.reshape(-1)
test_features = test_features.reshape(-1, test_features.shape[-1])
test_labels = test_labels.reshape(-1)

print(train_features.shape, test_features.shape,
      train_labels.shape, test_labels.shape)

train_features[:, :3] = np.exp(train_features[:, :3])
test_features[:, :3] = np.exp(test_features[:, :3])

lr = LogisticRegression(solver='liblinear', dual=True, max_iter=10000)
clf = RandomizedSearchCV(lr, {'C': np.random.uniform(low=0.001, high=5, size=(50000,))}, cv=3, n_jobs=-1).fit(
    train_features, train_labels)
print('Score:', clf.score(test_features, test_labels),
      clf.best_params_, clf.best_score_)
