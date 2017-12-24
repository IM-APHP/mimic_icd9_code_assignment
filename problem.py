# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from datetime import timedelta
from sklearn import model_selection

problem_title = 'Medical text classification in ICD 9 thesaurus'
_target_column_name = 'TARGET'
#Get the references from ICD thesaurus
ref_icd=pd.read_csv('./data/D_ICD_DIAGNOSES.csv',dtype={'ROW_ID':np.int32,
                                                      'ICD9_CODE': str,
                                                      'SHORT_TITLE': str,
                                                      'LONG_TITLE':  str})
#labels will be icd9 chapters
ref_icd['ICD9_CHAP']=ref_icd['ICD9_CODE'].str.slice(0,3)

_prediction_label_names = ref_icd['ICD9_CHAP'].unique()
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

soft_score_matrix = np.array(np.diag(np.ones(len(_prediction_label_names))))

true_false_score_matrix = np.array(np.diag(np.ones(len(_prediction_label_names))))


score_types = [
    rw.score_types.SoftAccuracy(
        name='sacc', score_matrix=soft_score_matrix, precision=3),
    rw.score_types.Accuracy(name='acc', precision=3),
    rw.score_types.SoftAccuracy(
        name='tfacc', score_matrix=true_false_score_matrix, precision=3),
]


def get_cv(X, y):
    """Slice folds by equal date intervals."""
    n_splits = 8
    for i in range(n_splits):
        train_is, test_is = model_selection.train_test_split(np.arange(len(y)))
        yield train_is, test_is

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), sep=',',
                          dtype={'HADM_ID':np.int32, 'TEXT':str, 'TARGET':str})
    # Re expand icd9 --> make one code by line
    data['TARGET']=data['TARGET'].apply(lambda x : eval(x))
    rows = []
    _ = data.apply(lambda row: [rows.append([row['HADM_ID'],row['TEXT'],nn]) 
                         for nn in row.TARGET], axis=1)
    data = pd.DataFrame(rows, columns=data.columns)
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
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