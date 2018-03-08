# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import fbeta_score
#Our own custom score

class Fbeta(rw.score_types.classifier_base.ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='Fbeta',beta=2):
        self.name = name
        self.beta = beta

    def __call__(self, y_true, y_pred):
        fbeta = fbeta_score(y_true, y_pred, average='weighted', beta=self.beta)
        return fbeta

problem_title = 'Medical text classification in ICD 9 thesaurus'

_prediction_label_names = [ '403', '048', '585', '425', '276', '724', '458', '287', '285',
       '275', '327', '338', '789', '790', '410', '414', '331', '530',
       '411', '482', '272', '305', '197', '424', '584', '682', '511',
       '599', '428', '401', '041', '571', '070', '250', '057', '572',
       '286', '518', '038', '280', '263', '303', '244', '112', '441',
       '049', '440', '274', '427', '569', '560', '491', '433', '043',
       '493', '416', '765', '076', '779', '774', '770', '362', '198',
       '780', '357', '293', '443', '031', '600', '294', '284', '553',
       '426', '707', '348', '787', '564', '300', '788', '453', '413',
       '507', '162', '785', '799', '574', '296', '733', '578', '438',
       '008', '593', '345', '519', '278', '715', '415', '535', '576',
       '288', '567', '786', '784', '729', '434', '456', '577', '562', '291']







#We make one binary class prediction for each code
predictions = []
for x in _prediction_label_names:
     predictions.append(rw.prediction_types.make_multiclass(
    label_names= ['0','1'] ))

#Combine the predictions
Predictions = rw.prediction_types.make_combined(predictions)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

score_types = []
#List to record the score we passed
score_f1 = []
score_acc = []
score_f2 = []
for i,pred in enumerate(predictions):
    #Add an f1 score and accuracy score
    score_f1.append(rw.score_types.F1Above(name = 'f1_' + str(i) , precision = 3 ))
    score_acc.append(rw.score_types.Accuracy(name = 'acc_' + str(i), precision = 3 ))
    score_f2.append(Fbeta(name = 'f2_' + str(i),beta=2))

#Each label has equal weights
weights = list(1/len(score_acc) * np.ones_like(score_acc))
score_types.append(rw.score_types.Combined(name = 'combined_accuracy' , score_types = score_acc,precision = 3 , weights = weights))
score_types.append(rw.score_types.Combined(name = 'combined_f1' , score_types = score_f1, precision = 3 ,weights = weights))
score_types.append(rw.score_types.Combined(name = 'f2' , score_types = score_f2, precision = 3 ,weights = weights))


def get_cv(X, y):
    n_splits = 8
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=57)
    return cv.split(X, y)



def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), sep=',',
                          dtype={'HADM_ID':np.int32, 'TEXT':str, 'TARGET':str})

    # Re expand icd9 -->  One hot encode the target
    mlb = MultiLabelBinarizer()

    data['TARGET'] = data['TARGET'].apply(lambda x : eval(x))
    temp = mlb.fit_transform(data['TARGET'])

    for i,x in enumerate(mlb.classes_):
        data[x] = temp[ : , i ]
    #No need for the target
    del data['TARGET']

    y_array = data[_prediction_label_names].values
    X_df = data.drop(_prediction_label_names, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[:100], y_array[:100]
    else:
        return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
