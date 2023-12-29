import os
dir(os)
os.chdir('Bachelor Project//2-Cat_Base_total')
os.listdir()

from tensorflow import keras

# !pip install pycaret==2.3.10 markupsafe==2.0.1 pyyaml==5.4.1 -qq
import tensorflow as tf
import pandas as pd
# import numpy as np
from datetime import datetime, timedelta
'''import warnings
warnings.filterwarnings('ignore')'''
# from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics  # (confusion_matrix)(accuracy_score)(recall_score)(precision_score)(f1_score)(classification_report)
import seaborn as sns
from imblearn.over_sampling import SMOTE



X_res = pd.read_csv('X_res_v2_max_min.csv', index_col=0)
y_res = pd.read_csv('y_res_v2_max_min.csv', index_col=0)

train_validation_merge = pd.read_csv('train_validation_merge_v2.csv', index_col=0)
test_validation_merge = pd.read_csv('test_validation_merge_v2.csv', index_col=0)

customer_ids = test_validation_merge.drop_duplicates('cid').sort_values('cid')
X_tr, X_ts, y_tr, y_ts = train_test_split(customer_ids, customer_ids['reorder_label'], random_state=42, test_size=0.3)
train_validation_merge = pd.concat(
    [train_validation_merge, test_validation_merge[test_validation_merge.cid.isin(X_tr.cid)]], ignore_index=True)
test_validation_merge = test_validation_merge[test_validation_merge.cid.isin(X_ts.cid)]

columns = [
    'days_since_prior_order',
    'day_of_week',
    'user_cat_reorder_rate',
                     'user_total_cats', 'user_avg_cart_size',
    'days_between_orders',

    'user_unique_cats', 'user_reordered_cats','user_ave_price_day_ratio','user_days_price_ratio_since_prior',
    'user_reordered_cats_ratio', 'user_unique_classes',
    'user_reordered_classes', 'user_reordered_classes_ratio',
                                 'cat_week_reorder_rate',
    'class_week_reorder_rate',
    't_days_since_prior_order_reorder_rate','ave_price_day_ratio',
    'c_days_since_prior_order_reorder_rate',
    'u_days_since_prior_order_reorder_rate',

    'u_t_days_since_prior_reorder_rate',
    'u_c_days_since_prior_reorder_rate', 'cat_reorder_rate',
    'p_reduced_feat_1', 'p_reduced_feat_2', 'p_reduced_feat_3',
    'u_t_order_rate', 'u_t_reorder_rate', 'u_t_orders_since_last',
     'u_c_orders_since_last', 'max_streak_cat']


len(columns)


# columns2= columns+['reorder_label']

def feature_target(data):
    global columns
    train_validation_feature = data[columns]

    train_validation_target = data[['reorder_label']]
    return train_validation_feature, train_validation_target


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
X_train, y_train = feature_target(train_validation_merge)
X_test, y_test = feature_target(test_validation_merge)

X_train_scaled = X_train/(X_train.max())
X_test_scaled = X_test/(X_train.max())



from imblearn.over_sampling import SMOTE

# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
#
# X_res.to_csv('X_res_v2.csv')
# y_res.to_csv('y_res_v2.csv')

# globals().keys()

del customer_ids, X_tr, X_ts, y_tr, y_ts, X_train, GradientBoostingClassifier, RandomForestClassifier, SMOTE,  test_validation_merge, train_validation_merge, X_test


def f1(prob=0.5):
    def threshold(row):
      if row > prob:
          return 1
      return 0
    y_pred = (pd.Series(y_pred_ann.reshape(-1))).apply(threshold)
    return metrics.f1_score(y_test, y_pred)
def best_params():
    best_treshold = 0.05
    best_f1 = 0.0001
    for i in range(19):
        if f1(0.05+i*0.05)>best_f1:
            best_f1 = f1(0.05+i*0.05)
            best_treshold = 0.05+i*0.05
    return best_treshold



import seaborn as sns
from sklearn import metrics

'''ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu', input_shape=(len(columns),)),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=8, activation='sigmoid'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])'''
import numpy as np
X_res=np.expand_dims(X_res, axis=2)
y_res=np.array(y_res)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)
y_test = np.array(y_test)
# (np.array(X_res)).shape
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.SimpleRNN(6,activation='sigmoid',use_bias=True))
ann.add(tf.keras.layers.Dropout(rate=0.05))
# ann.add(tf.keras.layers.Dense(units=3, activation='sigmoid'))
# ann.add(tf.keras.layers.Dense(units=5, activation='sigmoid'))
# ann.add(tf.keras.layers.Dense(units=5, activation='sigmoid'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
met = [tf.keras.metrics.PrecisionAtRecall(recall=0.78)]
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = met)

ann.fit(X_res, y_res, batch_size = 1024 , epochs = 100,verbose=1)
y_pred_trains = ann.predict(X_res)
y_pred_train = y_pred_trains >0.5
print(metrics.classification_report(y_pred_train, y_res))

y_pred_ann = ann.predict(X_test_scaled)
ss = y_pred_ann.copy()

y_pred_ann = (ss > best_params())

print(metrics.classification_report(y_test, y_pred_ann))
# sns.heatmap(metrics.confusion_matrix(y_test, y_pred_ann), square=True , annot=True, fmt='g')


met_str = list(str(x) for x in met)
os.chdir('NN_V2')
n=1
for i in os.listdir():
    if 'ANN_report_' in i:
        n+=1
report_path = f"RNN_report_{n}.txt"
text_file = open(report_path, "w")
# report = f"{ann.summary()}\n\n{metrics.classification_report(y_test, y_pred_ann)}\n\nmetric: {met_str}"
report = f"{metrics.classification_report(y_test, y_pred_ann)}\n\nmetric: {met_str}"
m = text_file.write(report)
text_file.close()

with open(f'RNN_summary_{n}.txt', 'w') as f:

    ann.summary(print_fn=lambda x: f.write(x + '\n'))
    