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



X_res = pd.read_csv('X_res_v3.csv', index_col=0)
y_res = pd.read_csv('y_res_v3.csv', index_col=0)

train_validation_merge = pd.read_csv('train_validation_merge_v3.csv', index_col=0)
test_validation_merge = pd.read_csv('test_validation_merge_v3.csv', index_col=0)

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
##################################################    ANN
# ann = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(units=32, activation='relu', input_shape=(len(columns),)),
#     tf.keras.layers.Dense(units=16, activation='relu'),
#     tf.keras.layers.Dense(units=8, activation='relu'),
#     tf.keras.layers.Dense(units=1, activation='sigmoid')
# ])
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=30, activation='relu'))
ann.add(tf.keras.layers.Dense(units=20, activation='relu'))
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=20, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
met = [tf.keras.metrics.PrecisionAtRecall(recall=0.70)]
ann.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics = met)
ann.fit(X_res, y_res, batch_size = 32 , epochs = 10,verbose=1)

y_pred_trains = ann.predict(X_res)
y_pred_train = y_pred_trains >0.5
print(metrics.classification_report(y_pred_train, y_res))

y_pred_ann = ann.predict(X_test_scaled)
ss = y_pred_ann.copy()

y_pred_ann = (ss > best_params())

print(metrics.classification_report(y_test, y_pred_ann))
# sns.heatmap(metrics.confusion_matrix(y_test, y_pred_ann), square=True , annot=True, fmt='g')
n=1
for i in os.listdir():
    if 'NN_V3//ANN_report_V3_' in i:
        n+=1


met_str = list(str(x) for x in met)
# n = 1+len(os.listdir())
report_path = f"NN_V3//ANN_report_V3_{n}.txt"
text_file = open(report_path, "w")
# report = f"{ann.summary()}\n\n{metrics.classification_report(y_test, y_pred_ann)}\n\nmetric: {met_str}"
report = f"{metrics.classification_report(y_test, y_pred_ann)}\n\nmetric: {met_str}"
m = text_file.write(report)
text_file.close()


with open(f'NN_V3//ANN_summary_V3_{n}.txt', 'w') as f:

    ann.summary(print_fn=lambda x: f.write(x + '\n'))

##################################################    CNN
import numpy as np

X_res = np.expand_dims(X_res, axis=2)
y_res = np.array(y_res)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)
y_test = np.array(y_test)
ann = tf.keras.models.Sequential()
X_res.shape
ann.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=X_res.shape))
ann.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
ann.add(tf.keras.layers.Dropout(0.5))
ann.add(tf.keras.layers.MaxPooling1D(pool_size=2))
ann.add(tf.keras.layers.Flatten())
ann.add(tf.keras.layers.Dense(100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
loss = tf.keras.losses.SparseCategoricalCrossentropy
met = [tf.keras.metrics.PrecisionAtRecall(recall=0.70)]
ann.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'binary_crossentropy', metrics = met)
ann.fit(X_res, y_res, batch_size = 32 , epochs = 30,verbose=1)

y_pred_trains = ann.predict(X_res)
y_pred_train = y_pred_trains >0.5
print(metrics.classification_report(y_pred_train, y_res))

y_pred_ann = ann.predict(X_test_scaled)
ss = y_pred_ann.copy()

y_pred_ann = (ss > best_params())

print(metrics.classification_report(y_test, y_pred_ann))
# sns.heatmap(metrics.confusion_matrix(y_test, y_pred_ann), square=True , annot=True, fmt='g')
n=1
for i in os.listdir():
    if 'NN_V3//CNN_report_V3_' in i:
        n+=1


met_str = list(str(x) for x in met)
# n = 1+len(os.listdir())
report_path = f"NN_V3//CNN_report_V3_{n}.txt"
text_file = open(report_path, "w")
# report = f"{ann.summary()}\n\n{metrics.classification_report(y_test, y_pred_ann)}\n\nmetric: {met_str}"
report = f"{metrics.classification_report(y_test, y_pred_ann)}\n\nmetric: {met_str}"
m = text_file.write(report)
text_file.close()


with open(f'NN_V3//CNN_summary_V3_{n}.txt', 'w') as f:

    ann.summary(print_fn=lambda x: f.write(x + '\n'))
##################################################    RNN
##################################################    RNN
##################################################    RNN
##################################################    RNN



ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu', input_shape=(len(columns),)),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
# ann = tf.keras.models.Sequential()
# ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=3, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
# ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
met = [tf.keras.metrics.PrecisionAtRecall(recall=0.47)]
ann.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics = met)
ann.fit(X_res, y_res, batch_size = 32 , epochs = 30,verbose=1)

y_pred_trains = ann.predict(X_res)
y_pred_train = y_pred_trains >0.5
print(metrics.classification_report(y_pred_train, y_res))

y_pred_ann = ann.predict(X_test_scaled)
ss = y_pred_ann.copy()

y_pred_ann = (ss > best_params())

print(metrics.classification_report(y_test, y_pred_ann))
# sns.heatmap(metrics.confusion_matrix(y_test, y_pred_ann), square=True , annot=True, fmt='g')
n=1
for i in os.listdir():
    if 'NN_V3//RNN_report_V3_' in i:
        n+=1


met_str = list(str(x) for x in met)
# n = 1+len(os.listdir())
report_path = f"NN_V3//RNN_report_V3_{n}.txt"
text_file = open(report_path, "w")
# report = f"{ann.summary()}\n\n{metrics.classification_report(y_test, y_pred_ann)}\n\nmetric: {met_str}"
report = f"{metrics.classification_report(y_test, y_pred_ann)}\n\nmetric: {met_str}"
m = text_file.write(report)
text_file.close()


with open(f'NN_V3//RNN_summary_V3_{n}.txt', 'w') as f:

    ann.summary(print_fn=lambda x: f.write(x + '\n'))




'''def plot_classification_report(kind ,model_name, prob=0.5):
  if kind=='prob':
    def threshold(row):
      if row > prob:
          return 1
      return 0
    y_pred = y_df['y_test_prob'].apply(threshold)
    report = metrics.classification_report(y_df['y_test'], y_pred)
    print(report)
    # classes = ['class 1', 'class 2', 'class 3']
    report_path = f"CR_CB_tot\{model_name}.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report)
    text_file.close()
    plt.figure(figsize=(8, 6), dpi=80)
    sns.heatmap(metrics.confusion_matrix(y_df['y_test'], y_df['y_test_prediction']), annot=True, fmt='g')
    plt.savefig(f'CR_CB_tot\{model_name}.png')
    plt.show()
    return metrics.f1_score
  elif kind=='pred':
    report = metrics.classification_report(y_df['y_test'], y_df['y_test_prediction'])
    print(report)
    # classes = ['class 1', 'class 2', 'class 3']
    report_path = f"CR_CB_tot\{model_name}.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report)
    text_file.close()
    plt.figure(figsize=(8, 6), dpi=80)
    sns.heatmap(metrics.confusion_matrix(y_df['y_test'], y_df['y_test_prediction']), annot=True, fmt='g')
    plt.savefig(f'CR_CB_tot\{model_name}.png')
    plt.show()
    return metrics.f1_score
  else:
    print('Invalid kind')


def f1(kind, prob=0.5):
  if kind=='prob':
    def threshold(row):
      if row > prob:
          return 1
      return 0
    y_pred = y_df['y_test_prob'].apply(threshold)
    return metrics.f1_score(y_df['y_test'], y_pred)
  elif kind=='pred':
    return metrics.f1_score(y_df['y_test'], y_df['y_test_prediction'])
  else:
    print('Invalid kind')

def best_params():
    best_kind = 'prob'
    best_treshold = 0.05
    best_f1 = 0.0001
    for i in range(19):
        if f1('prob',0.05+i*0.05)>best_f1:
            best_f1 = f1('prob',0.05+i*0.05)
            best_treshold = 0.05+i*0.05
    if f1('pred')>f1('prob',best_treshold):
        best_kind = 'pred'
    return best_kind,best_treshold
'''



'''
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

# define hyperparameter space to search over
param_grid = {
    'hidden_layer_sizes': [(10,),(10,10),(10,10,10),(20,),(20,20),(20,20,20),(5,),(5,5),(5,5,5),(10,20),(5,10,20),(20,10),(20,10,5),(25,15),(30,20,10),(10,20,30)],
    'activation': ['relu', 'tanh','identity','logistic'],
    'solver': ['sgd','adam'],
    'learning_rate': ['constant', 'adaptive','invscaling'],
}

# initialize the model
model = MLPClassifier()

# define the grid search

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')

# fit the grid search to the data
start_time = datetime.now()
print("Training Started :")
grid_search.fit(X_res, y_res)
print("Training Completed ")
end_time = datetime.now()
difference = end_time - start_time
print(difference)

# evaluate the best model on the test set
y_pred = grid_search.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred)
print(f1)
print(grid_search.best_params_)


text_file = open("ANN_best_params.txt", "w")
n = text_file.write(grid_search.best_params_)
text_file.close()
































# fookin ANN

from sklearn.neural_network import MLPClassifier

# output Training Time
start_time = datetime.now()
print("Training Started :")
xgbc = MLPClassifier()
xgbc.fit(X_res, y_res)
print("Training Completed ")
end_time = datetime.now()
difference = end_time - start_time
print(difference)

y_test_prediction = xgbc.predict(X_test_scaled)
y_test_prob = xgbc.predict_proba(X_test_scaled)
y_df = pd.DataFrame({'y_test': y_test['reorder_label'],
                     'y_test_prob': y_test_prob[:, 1],
                     'y_test_prediction': y_test_prediction})
# # train analysis
# y_train_prediction = xgbc.predict(X_train_scaled)
# y_train_prob = xgbc.predict_proba(X_train_scaled)
# y_train_df = pd.DataFrame({'y_train': y_train['reorder_label'],
#                            'y_train_prob': y_train_prob[:, 1],
#                            'y_train_prediction': y_train_prediction})
importance = xgbc.feature_importances_
# plot feature importance
plt.figure(figsize=(11, 6), dpi=80)
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.52, right=0.96, top=0.96)
plt.bar(columns, importance)
plt.xticks(rotation=3 * 90)
plt.savefig('FI_CB_tot\FI_ANN.JPEG', dpi=300)
plt.show()

ANN_k,ANN_t = best_params()
plot_classification_report(ANN_k, 'ANN',ANN_t)
del MLPClassifier, xgbc, importance, y_test_prediction, y_test_prob
'''


