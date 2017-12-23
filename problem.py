# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from datetime import timedelta
from sklearn import model_selection

problem_title = 'Medical text classification in ICD 9 thesaurus'
_target_column_name = 'TARGET'
_prediction_label_names = ['00845','0311','038','0389','041','0412','0431','0486','0496','0570','070','07054','0769','112','162','197','198','2449','250','25000','25060','2639','2720','2724','2749','275','276','2760','2761','2762','2765','27651','27652','2767','2768','27800','27801','2800','2809','284','2851','28521','28529','2859','286','2869','2875','288','291','2930','2948','296','30000','3004','303','305','3051','32723','331','338','345','348','3572','362','4019','40390','40391','410','41071','4111','4139','414','41400','41401','415','4168','4240','4241','4254','426','427','4271','42731','42732','4275','42789','428','4280','42822','42823','42832','42833','433','434','438','440','441','4439','453','456','45829','4589','482','49121','493','49390','5070','5119','518','5180','5185','51881','519','530','53081','535','553','5601','562','564','567','569','5712','5715','572','5723','574','576','577','578','5789','5845','5849','585','5856','5859','593','599','5990','60000','682','707','70703','715','724','729','733','73300','765','770','77081','7742','779','7793','780','78039','784','785','78551','78552','786','787','78791','788','78820','789','790','7907','79902']
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

soft_score_matrix = np.array([
    [1, 0.8, 0, 0, 0, 0],
    [0.4, 1, 0.4, 0, 0, 0],
    [0, 0.4, 1, 0.4, 0, 0],
    [0, 0, 0.4, 1, 0.4, 0],
    [0, 0, 0, 0.4, 1, 0.4],
    [0, 0, 0, 0, 0.8, 1],
])

true_false_score_matrix = np.array([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
])

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
    data = pd.read_csv(os.path.join(path, 'data', f_name), sep='\t')
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
