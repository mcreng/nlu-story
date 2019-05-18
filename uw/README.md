# UW NLP Approach

Approach done following [Story Cloze Task: UW NLP System](https://roys174.github.io//papers/language_constraint/lsdsem_uw_nlp.pdf). Paper claimed to have 75% accuracy, this implementation can achieve 65%.

The files are:
* `lstm_train.py`: Training script of a LSTM LM.
* `lstm_eval.py`: Evaluating the trained LSTM LM to obtain probabilities of sentences.
* `preprocess.py`: Preprocessing training data and obtaining feature vectors.
* `lr.py`: Logistic Regression classifier on constructed feature vectors.