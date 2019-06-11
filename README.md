# BiLSTM with Attention and Dropout

Approach done following [Pay Attention to the Ending:
Strong Neural Baselines for the ROC Story Cloze Task](https://aclanthology.info/papers/P17-2097/p17-2097). Paper claimed to have 81.24% accuracy, this implementation can achieve 78.40%.

The files are:
* `biLSTM_att_val_ensem.ipynb`: Jupyter notebook for training ensemble model (with dropout layer)
* `biLISTM_Att_augTrainSet.ipynb`: Jupyter notebook for training attention only data (on augmented data)
* `biLSTM_Att_augTrainSet.py`: python file for training attention only data (on augmented data)
* `biLSTM_Att_val.py`: python file for training attention only data (on validation data)
* `biLSTM_att_val_ensem.py`:python file for training ensembled only data (on validation data only or on validation data + test data) (with test included)  

Link to parameters which is choosen to output final result: https://polybox.ethz.ch/index.php/s/GgkHdN1gJeMje8M

Notice of loading trained parameters: 

* We originally trained 10 models, and we validate on every 9 of them. The best performed model set is choosen, and in this case, its the one with the 4th model excluded has the best performance.

* To load the trained parameter, the biLSTM_att_val_ensem.py or biLSTM_att_val_ensem.ipynb should be run.
