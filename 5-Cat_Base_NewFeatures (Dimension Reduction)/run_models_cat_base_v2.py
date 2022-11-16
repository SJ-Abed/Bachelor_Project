

# !pip install pycaret==2.3.10 markupsafe==2.0.1 pyyaml==5.4.1 -qq

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
'''import warnings
warnings.filterwarnings('ignore')'''
# from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
# from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn import metrics  # (confusion_matrix)(accuracy_score)(recall_score)(precision_score)(f1_score)(classification_report)
import seaborn as sns
from imblearn.over_sampling import SMOTE


#read normalized and balanced train data
X_res = pd.read_csv('X_res_v2_2.csv', index_col=0)
y_res = pd.read_csv('y_res_v2_2.csv', index_col=0)

#read actual data
train_validation_merge = pd.read_csv('train_validation_merge_v2.2.csv', index_col=0)
test_validation_merge = pd.read_csv('test_validation_merge_v2.2.csv', index_col=0)

# train-test-split
customer_ids = test_validation_merge.drop_duplicates('cid').sort_values('cid')
X_tr, X_ts, y_tr, y_ts = train_test_split(customer_ids, customer_ids['reorder_label'], random_state=42, test_size=0.3)
train_validation_merge = pd.concat(
    [train_validation_merge, test_validation_merge[test_validation_merge.cid.isin(X_tr.cid)]], ignore_index=True)
test_validation_merge = test_validation_merge[test_validation_merge.cid.isin(X_ts.cid)]

#for selecting feature
columns = ['days_since_prior_order', 'day_of_week','user_cat_reorder_rate', 'user_total_cats', 'user_avg_cart_size',
       'days_between_orders', 'user_unique_cats', 'user_reordered_cats',
       'user_reordered_cats_ratio', 'user_unique_classes',
       'user_reordered_classes', 'user_reordered_classes_ratio',
       'cat_week_reorder_rate', 'class_week_reorder_rate',
       't_days_since_prior_order_reorder_rate',
       'c_days_since_prior_order_reorder_rate',
       'u_days_since_prior_order_reorder_rate',
       'u_t_days_since_prior_reorder_rate',
       'u_c_days_since_prior_reorder_rate', 'ave_price_day_ratio',
       'days_between_cat_orders', 'cat_reorder_rate', 'p_reduced_feat_1',
       'p_reduced_feat_2', 'p_reduced_feat_3', 'u_t_order_rate',
       'u_t_reorder_rate', 'user_ave_price_day_ratio',
       'days_between_user_cat_orders', 'user_days_price_ratio_since_prior',
       'u_t_orders_since_last', 'u_c_orders_since_last', 'max_streak_cat',
       'since_prior_days_ratio', 'since_prior_days_cat_ratio',
       'since_prior_days_user_cat_ratio', 'user_unique_cat_ratio',
       'user_DPR_user_ratio', 'user_DPR_tot_ratio']
len(columns)



# columns2= columns+['reorder_label']

#split label of every row from its features
def feature_target(data):
    global columns
    train_validation_feature = data[columns]

    train_validation_target = data[['reorder_label']]
    return train_validation_feature, train_validation_target


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, y_train = feature_target(train_validation_merge)
X_test, y_test = feature_target(test_validation_merge)

#normalize data
scaler = StandardScaler().fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=columns)

'''params = {'n_estimators': 300,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01}'''

from imblearn.over_sampling import SMOTE

#balance data amd save them
sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
# X_res.to_csv('X_res_v2_2.csv')
# y_res.to_csv('y_res_v2_2.csv')

del customer_ids, X_tr, X_ts, y_tr, y_ts, X_train, GradientBoostingClassifier, RandomForestClassifier, SMOTE, scaler, sm, test_validation_merge, train_validation_merge, X_test

import seaborn as sns
from sklearn import metrics


#plot classification report and save result
def plot_classification_report(kind ,model_name, prob=0.5):
  if kind=='prob':
    def threshold(row):
      if row > prob:
          return 1
      return 0
    y_pred = y_df['y_test_prob'].apply(threshold)
    report = metrics.classification_report(y_df['y_test'], y_pred)
    print(report)
    # classes = ['class 1', 'class 2', 'class 3']
    report_path = f"CR_CB_tot2\{model_name}.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report)
    text_file.close()
    plt.figure(figsize=(8, 6), dpi=80)
    sns.heatmap(metrics.confusion_matrix(y_df['y_test'], y_df['y_test_prediction']), annot=True, fmt='g')
    plt.savefig(f'CR_CB_tot2\{model_name}.png')
    plt.show()
    return metrics.f1_score
  elif kind=='pred':
    report = metrics.classification_report(y_df['y_test'], y_df['y_test_prediction'])
    print(report)
    # classes = ['class 1', 'class 2', 'class 3']
    report_path = f"CR_CB_tot2\{model_name}.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report)
    text_file.close()
    plt.figure(figsize=(8, 6), dpi=80)
    sns.heatmap(metrics.confusion_matrix(y_df['y_test'], y_df['y_test_prediction']), annot=True, fmt='g')
    plt.savefig(f'CR_CB_tot2\{model_name}.png')
    plt.show()
    return metrics.f1_score
  else:
    print('Invalid kind')

#find best treshold for maximazing f1-score
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

# pycaret_df=X_res.__deepcopy__()
# pycaret_df['reorder_label'] = y_res
# pycaret_setup = setup(data = pycaret_train,target = 'reorder_label',normalize = True,
#             pca=False,silent = True,session_id = 3650, numeric_features =numeric_col)

# Logistic regression

# logistic regression for feature importance
from sklearn.linear_model import LogisticRegression

# define the model
model = LogisticRegression()
# fit the model
start_time = datetime.now()
print("Training Started :")
model.fit(X_res, y_res)
print("Training Completed ")
end_time = datetime.now()
difference = end_time - start_time
print(difference)
# get importance
importance = model.coef_[0]
# summarize feature importance
# for i,v in enumerate(importance):
# 	print('Feature: %s, Score:\t\t %.5f' % (columns[i],v))
# predict test data
y_test_prediction = model.predict(X_test_scaled)
y_test_prob = model.predict_proba(X_test_scaled)
y_df = pd.DataFrame({'y_test': y_test['reorder_label'],
                     'y_test_prob': y_test_prob[:, 1],
                     'y_test_prediction': y_test_prediction})
# train analysis
y_train_prediction = model.predict(X_train_scaled)
y_train_prob = model.predict_proba(X_train_scaled)
y_train_df = pd.DataFrame({'y_train': y_train['reorder_label'],
                           'y_train_prob': y_train_prob[:, 1],
                           'y_train_prediction': y_train_prediction})

# plot feature importance
plt.figure(figsize=(11, 7), dpi=80)
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.52, right=0.96, top=0.96)
plt.bar(columns, abs(importance))
plt.xticks(rotation=3 * 90)
plt.savefig('FI_CB_tot2\FI_LogR.JPEG', dpi=300)
plt.show()

log_k,log_t = best_params()

plot_classification_report(log_k, 'logR', log_t)

del LogisticRegression, model, importance, difference, end_time, start_time, y_df, y_test_prediction, y_train_prediction


# Naive Bayes

# permutation feature importance with knn for classification
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB

# define the model
model = GaussianNB()
# fit the model
start_time = datetime.now()
print("Training Started :")
model.fit(X_res, y_res)
print("Training Completed ")
end_time = datetime.now()
difference = end_time - start_time
print(difference)

from sklearn.inspection import permutation_importance

# perform permutation importance
results = permutation_importance(model, X_res, y_res, scoring='accuracy')
# get importance
importance = results.importances_mean
# summarize feature importance
# for i,v in enumerate(importance):
# 	print('Feature: %s, Score:\t\t %.5f' % (columns[i],v))


# predict test data
y_test_prediction = model.predict(X_test_scaled)
y_test_prob = model.predict_proba(X_test_scaled)
y_df = pd.DataFrame({'y_test': y_test['reorder_label'],
                     'y_test_prob': y_test_prob[:, 1],
                     'y_test_prediction': y_test_prediction})
# train analysis
y_train_prediction = model.predict(X_train_scaled)
y_train_prob = model.predict_proba(X_train_scaled)
y_train_df = pd.DataFrame({'y_train': y_train['reorder_label'],
                           'y_train_prob': y_train_prob[:, 1],
                           'y_train_prediction': y_train_prediction})

# plot feature importance
plt.figure(figsize=(11, 6), dpi=80)
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.52, right=0.96, top=0.96)
plt.bar(columns, abs(importance))
plt.xticks(rotation=3 * 90)
plt.savefig('FI_CB_tot2\FI_GNB2.JPEG', dpi=300)
plt.show()

NB_k,NB_t = best_params()
plot_classification_report(NB_k, 'GNB', NB_t)
del GaussianNB, model, importance, results, y_test_prediction, y_train_prediction, y_train_prob, y_test_prob



# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# output Training Time
start_time = datetime.now()
print("Training Started :")
xgbc = DecisionTreeClassifier()
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
# train analysis
y_train_prediction = xgbc.predict(X_train_scaled)
y_train_prob = xgbc.predict_proba(X_train_scaled)
y_train_df = pd.DataFrame({'y_train': y_train['reorder_label'],
                           'y_train_prob': y_train_prob[:, 1],
                           'y_train_prediction': y_train_prediction})
importance = xgbc.feature_importances_
# plot feature importance
plt.figure(figsize=(11, 6), dpi=80)
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.52, right=0.96, top=0.96)
plt.bar(columns, importance)
plt.xticks(rotation=3 * 90)
plt.savefig('FI_CB_tot2\FI_DT.JPEG', dpi=300)
plt.show()

DT_k,DT_t = best_params()
plot_classification_report(DT_k, 'DT',DT_t)

del DecisionTreeClassifier, xgbc, importance, y_test_prediction, y_train_prediction, y_train_prob, y_test_prob


# Random Forest
from sklearn.ensemble import RandomForestClassifier

# output Training Time
start_time = datetime.now()
print("Training Started :")
xgbc = RandomForestClassifier()
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
# train analysis
y_train_prediction = xgbc.predict(X_train_scaled)
y_train_prob = xgbc.predict_proba(X_train_scaled)
y_train_df = pd.DataFrame({'y_train': y_train['reorder_label'],
                           'y_train_prob': y_train_prob[:, 1],
                           'y_train_prediction': y_train_prediction})
importance = xgbc.feature_importances_
# plot feature importance
plt.figure(figsize=(11, 6), dpi=80)
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.52, right=0.96, top=0.96)
plt.bar(columns, importance)
plt.xticks(rotation=3 * 90)
plt.savefig('FI_CB_tot2\FI_RF.JPEG', dpi=300)
plt.show()


RF_k,RF_t = best_params()
plot_classification_report(RF_k, 'RF',RF_t)

del RandomForestClassifier, xgbc, importance, y_test_prediction, y_train_prediction, y_train_prob, y_test_prob

# import pickle
# filename = 'finalized_model_item_base_RF.sav'
# pickle.dump(xgbc, open(filename, 'wb'))


# Extra Trees

from sklearn.ensemble import ExtraTreesClassifier

# output Training Time
start_time = datetime.now()
print("Training Started :")
xgbc = ExtraTreesClassifier()
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
plt.savefig('FI_CB_tot2\FI_ET.JPEG', dpi=300)
plt.show()

ET_k,ET_t = best_params()
plot_classification_report(ET_k, 'ET',ET_t)
del ExtraTreesClassifier, xgbc, importance, y_test_prediction, y_test_prob



# Gradient Boosting

# Commented out IPython magic to ensure Python compatibility.
# %%html
# <!DOCTYPE html><html><body align="right"><br><div style="direction:;ltr"><p style="background-color: #F7FFCA; color:black; border:1px solid black; border-radius: 10px; font-size:36px; line-height:1.8; font-family: byekan; text-align:justify; padding-left: 50px; padding-right: 50px; padding: 25px;" >
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

# output Training Time
start_time = datetime.now()
print("Training Started :")
xgbc = AdaBoostClassifier()
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
plt.savefig('FI_CB_tot2\FI_ADA.JPEG', dpi=300)
plt.show()

ADA_k,ADA_t = best_params()
plot_classification_report(ADA_k,'ADA', ADA_t)


# XGBoost

# xgboost for feature importance on a classification problem
from xgboost import XGBClassifier
from matplotlib import pyplot

# output Training Time
start_time = datetime.now()
print("Training Started :")
model = XGBClassifier()
model.fit(X_res, y_res)
print("Training Completed ")
end_time = datetime.now()
difference = end_time - start_time
print(difference)

y_test_prediction = model.predict(X_test_scaled)
y_test_prob = model.predict_proba(X_test_scaled)
y_df = pd.DataFrame({'y_test': y_test['reorder_label'],
                     'y_test_prob': y_test_prob[:, 1],
                     'y_test_prediction': y_test_prediction})
# # train analysis
# y_train_prediction = model.predict(X_train_scaled)
# y_train_prob = model.predict_proba(X_train_scaled)
# y_train_df = pd.DataFrame({'y_train': y_train['reorder_label'],
#                            'y_train_prob': y_train_prob[:, 1],
#                            'y_train_prediction': y_train_prediction})
importance = model.feature_importances_
# plot feature importance
plt.figure(figsize=(11, 6), dpi=80)
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.52, right=0.96, top=0.96)
plt.bar(columns, importance)
plt.xticks(rotation=3 * 90)
plt.savefig('FI_CB_tot2\FI_XGB.JPEG', dpi=300)
plt.show()
XGB_k,XGB_t = best_params()
plot_classification_report(XGB_k,'XGB', XGB_t)




#reduced features accourding to feature Importance plot of XGB model:
'''columns = ['days_since_prior_order', 'day_of_week', 'user_avg_cart_size',
       'user_unique_classes',
       'user_reordered_classes',  'class_week_reorder_rate',
       't_days_since_prior_order_reorder_rate',
       'days_between_cat_orders', 'cat_reorder_rate',
       'u_t_reorder_rate',
       'u_t_orders_since_last', 'u_c_orders_since_last', 'max_streak_cat',
       'since_prior_days_user_cat_ratio',
       'user_DPR_user_ratio']
X_res =X_res[columns]'''

from xgboost import XGBClassifier
# output Training Time
start_time = datetime.now()
print("Training Started :")
model = XGBClassifier()
model.fit(X_res, y_res)
print("Training Completed ")
end_time = datetime.now()
difference = end_time - start_time
print(difference)

y_test_prediction = model.predict(X_test_scaled)
y_test_prob = model.predict_proba(X_test_scaled)
y_df = pd.DataFrame({'y_test': y_test['reorder_label'],
                     'y_test_prob': y_test_prob[:, 1],
                     'y_test_prediction': y_test_prediction})

importance = model.feature_importances_
# plot feature importance
plt.figure(figsize=(11, 6), dpi=80)
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.52, right=0.96, top=0.96)
plt.bar(columns, importance)
plt.xticks(rotation=3 * 90)
plt.savefig('FI_CB_tot2\FI_XGB_DR.JPEG', dpi=300)
plt.show()
XGB_k,XGB_t = best_params()
plot_classification_report(XGB_k,'XGB_DR', XGB_t)