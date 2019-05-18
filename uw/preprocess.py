import pickle
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy

root = './gdrive/My Drive/Colab Notebooks/'

# load LM
filenames = {
    'tokenizer': root+'tokenizer.pkl',
    'dataloader': root+'dataloader.pkl',
    'model': root+'weights.10-2.62.hdf5'
}
lm_eval = LMEvaluator(**filenames)

# load tagger
nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])

# load training data
df = pd.read_csv(root+'eval_stories.csv')
# df = pd.read_csv('./test.csv')

# split training and validation 9:1
n_stories = len(df)
print(int(np.ceil(n_stories*.9)))
train = {
    'raw': df.iloc[:int(np.ceil(n_stories*.9))]
}
print(len(train['raw']))
test = {
    'raw': df.iloc[int(np.ceil(n_stories*.9)):].reset_index()
}
print(len(test['raw']))

# preprocess data
for d in [train, test]:
    d['stories'] = d['raw'].apply(lambda x: x[-7]+' '+x[-6]+' ' +
                                  x[-5]+' '+x[-4], axis=1)
    d['full1'] = d['raw'].apply(lambda x: x[-7]+' '+x[-6]+' ' +
                                x[-5]+' '+x[-4]+' '+x[-3], axis=1)
    d['full2'] = d['raw'].apply(lambda x: x[-7]+' '+x[-6]+' ' +
                                x[-5]+' '+x[-4]+' '+x[-2], axis=1)
    d['endings'] = d['raw'].iloc[:, [-3, -2]]

    d['tokens'] = d['endings'].applymap(nlp)
    d['ptokens'] = d['tokens'].copy()
    d['pos'] = d['tokens'].applymap(
        lambda doc: list(map(lambda t: t.pos_, doc)))
    d['labels'] = d['raw'].iloc[:, -1]
    
print('hi')

# count tokens, change infrequent ones to pos tags
word_cv = CountVectorizer(ngram_range=(1, 1), min_df=1)
X = word_cv.fit_transform(
    train['endings'].values.reshape(-1).tolist()).A
X = X.reshape(-1, 2, X.shape[-1])  # shape (N, 2, n_features)
C = np.sum(X, axis=(0, 1))
stop_words = word_cv.inverse_transform(np.where(C <= 5)[0])[0]

for d in [train, test]:
    for r_idx, docs in d['tokens'].iterrows():
        for d_idx, doc in enumerate(docs):
            d['ptokens'].iloc[r_idx, d_idx] = ' '.join([t.text if t.text.lower(
            ) not in stop_words else d['pos'].iloc[r_idx, d_idx][idx] for idx, t in enumerate(doc)])
print('hi')
            
# word-level ngram feature
word_cv = TfidfVectorizer(ngram_range=(1, 5), min_df=5, lowercase=False)
train['word_X'] = word_cv.fit_transform(
    train['ptokens'].values.reshape(-1).tolist()).A
train['word_X'] = train['word_X'].reshape(-1, 2, train['word_X'].shape[-1])
test['word_X'] = word_cv.transform(
    test['ptokens'].values.reshape(-1).tolist()).A
test['word_X'] = test['word_X'].reshape(-1, 2, test['word_X'].shape[-1])

print('hi')

# char-level ngram feature
char_cv = TfidfVectorizer(ngram_range=(4, 4), min_df=5, analyzer='char')
train['char_X'] = char_cv.fit_transform(
    train['endings'].values.reshape(-1).tolist()).A
train['char_X'] = train['char_X'].reshape(-1, 2, train['char_X'] .shape[-1])
test['char_X'] = char_cv.transform(
    test['endings'].values.reshape(-1).tolist()).A
test['char_X'] = test['char_X'].reshape(-1, 2, test['char_X'] .shape[-1])

print('hi')

# LM features
for d in [train, test]:
    d['p_ending'] = d['endings'].applymap(lambda s: lm_eval.evaluate([s])[0]).values[:, :, None]
#     d['p_ending'] = np.array(lm_eval.evaluate(
#         d['endings'].values.reshape(-1).tolist())).reshape(-1, 2, 1)
    d['p_cond'] = np.vstack([d['full1'].apply(lambda s: lm_eval.evaluate([s])[0]).values, d['full2'].apply(lambda s: lm_eval.evaluate([s])[0]).values]) - d['stories'].apply(lambda s: lm_eval.evaluate([s])[0]).values
    d['p_cond'] = d['p_cond'].T[:, :, None]
    d['p_ratio'] = d['p_ending'] - d['p_cond']

print('hi')
    
# length feature
for d in [train, test]:
    d['len'] = d['endings'].applymap(len).values[:, :, None]

print(train['p_ending'].shape)
print(train['p_cond'].shape)
print(train['p_ratio'].shape)
print(train['len'].shape)
print(train['word_X'].shape)
print(train['char_X'].shape)

print(test['p_ending'].shape)
print(test['p_cond'].shape)
print(test['p_ratio'].shape)
print(test['len'].shape)
print(test['word_X'].shape)
print(test['char_X'].shape)

# concat features and save
for d in [train, test]:
    d['features'] = np.concatenate([
        d['p_ending'],
        d['p_cond'],
        d['p_ratio'],
        d['len'],
        d['word_X'],
        d['char_X']
    ], axis=-1)
    print(d['features'].shape)

with open(root+'train_features.pkl', 'wb') as f:
    pickle.dump(train['features'], f)
with open(root+'train_labels.pkl', 'wb') as f:
    pickle.dump(train['labels'], f)
with open(root+'test_features.pkl', 'wb') as f:
    pickle.dump(test['features'], f)
with open(root+'test_labels.pkl', 'wb') as f:
    pickle.dump(test['labels'], f)