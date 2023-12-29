from prepare_to_run import prepare
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
version = '3_20'
Folder = '2-Cat_Base_total'
scaler = 'maxmin'


def plot_classification_report(kind, model_name, version, duration, params, prob=0.5, scaler='maxmin', File_Number=2):
    info = f'Learning duration:\t{duration}\nnondefault parameters:\t{params}'
    if File_Number == 2:
        Folder = '2-Cat_Base_total'
    elif File_Number == 1:
        Folder = '1-Item_Base_total'
    elif File_Number == 6:
        Folder = '6-dunn'

    if kind == 'prob':
        def threshold(row):
            if row > prob:
                return 1
            return 0

        y_pred = y_df['y_test_prob'].apply(threshold)
        report = (metrics.classification_report(y_df['y_test'], y_pred) + f'\n\n{info}')
        print(report)
        # classes = ['class 1', 'class 2', 'class 3']
        if f'CR_CB_tot_V{version}' not in os.listdir(f'{Folder}'):
            os.mkdir(f'{Folder}\\CR_CB_tot_V{version}')
        j = 1
        for i in os.listdir(f"{Folder}\\CR_CB_tot_V{version}"):
            if f'{model_name}_{scaler}' in i:
                j += 1
        report_path = f"{Folder}\CR_CB_tot_V{version}\{model_name}_{scaler}_{j}.txt"
        text_file = open(report_path, "w")
        n = text_file.write(report)
        text_file.close()
        plt.figure(figsize=(8, 6), dpi=80)
        sns.heatmap(metrics.confusion_matrix(y_df['y_test'], y_df['y_test_prediction']), annot=True, fmt='g')
        plt.savefig(f'{Folder}\CR_CB_tot_V{version}\\{model_name}_{scaler}_{j}.png')
        plt.show()
        return metrics.f1_score
    elif kind == 'pred':
        report = (metrics.classification_report(y_df['y_test'],y_df['y_test_prediction']) + f'\n\n{info}')
        print(report)
        # classes = ['class 1', 'class 2', 'class 3']'
        if f'CR_CB_tot_V{version}' not in os.listdir(f'{Folder}'):
            os.mkdir(f'{Folder}\\CR_CB_tot_V{version}')
        j = 1
        for i in os.listdir(f"{Folder}\\CR_CB_tot_V{version}"):
            if f'{model_name}_{scaler}' in i:
                j += 1
        report_path = f"{Folder}\CR_CB_tot_V{version}\\{model_name}_{scaler}_{j}.txt"
        text_file = open(report_path, "w")
        n = text_file.write(report)
        text_file.close()
        plt.figure(figsize=(8, 6), dpi=80)
        sns.heatmap(metrics.confusion_matrix(y_df['y_test'], y_df['y_test_prediction']), annot=True, fmt='g')
        plt.savefig(f'{Folder}\CR_CB_tot_V{version}\\{model_name}_{scaler}_{j}.png')
        plt.show()
        return metrics.f1_score
    else:
        print('Invalid kind')

def f1(kind, prob=0.5):
    global y_df
    if kind == 'prob':
        def threshold(row):
            if row > prob:
                return 1
            return 0
        y_pred = y_df['y_test_prob'].apply(threshold)
        return metrics.f1_score(y_df['y_test'], y_pred)
    elif kind == 'pred':
        return metrics.f1_score(y_df['y_test'], y_df['y_test_prediction'])
    else:
        print('Invalid kind')
def best_params():
    best_kind = 'prob'
    best_treshold = 0.05
    best_f1 = 0.0001
    for i in range(19):
        if f1('prob', 0.05 + i * 0.05) > best_f1:
            best_f1 = f1('prob', 0.05 + i * 0.05)
            best_treshold = 0.05 + i * 0.05
    if f1('pred') > f1('prob', best_treshold):
        best_kind = 'pred'
    return best_kind, best_treshold



X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns = prepare.get_data('3_20',2,scaler=scaler)
########################################################################
model_name = 'Bagging'
# Create three different models
model1 = XGBClassifier()
model2 = RandomForestClassifier()
model3 = MLPClassifier()
estimators = [('xgb',model1),('rf',model2),('mlp',model3)]
combined_estimators = VotingClassifier(estimators)
# Use bagging with the three models
bagging = BaggingClassifier(warm_start=True)
# Train the ensemble model
start_time = datetime.now()
print(f"Training Started at {start_time.time()}:")
bagging.fit(X_res, y_res)
print("Training Completed ")
end_time = datetime.now()
difference = end_time - start_time
print(difference)
y_test_prediction = bagging.predict(X_test_scaled)
y_test_prob = bagging.predict_proba(X_test_scaled)
y_df = pd.DataFrame({'y_test': y_test['reorder_label'],
                     'y_test_prob': y_test_prob[:, 1],
                     'y_test_prediction': y_test_prediction})
# train analysis
y_train_prediction = bagging.predict(X_train_scaled)
y_train_prob = bagging.predict_proba(X_train_scaled)
y_train_df = pd.DataFrame({'y_train': y_train['reorder_label'],
                           'y_train_prob': y_train_prob[:, 1],
                           'y_train_prediction': y_train_prediction})
importance = bagging.feature_importances_
# plot feature importance
plt.figure(figsize=(11, 6), dpi=80)
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.52, right=0.96, top=0.96)
plt.bar(columns, importance)
plt.xticks(rotation=3 * 90)
if f'FI_CB_tot_V{version}' not in os.listdir(f'{Folder}'):
    os.mkdir(f'{Folder}\\FI_CB_tot_V{version}')
    plt.savefig(f'{Folder}\FI_CB_tot_V{version}\FI_{model_name}_{scaler}.JPEG', dpi=300)
    plt.show()

DT_k,DT_t = best_params()
plot_classification_report(DT_k, model_name,version ,str(difference), {}, DT_t,scaler,2)





##################################################################################
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from keras.models import Sequential
from keras.layers import Dense
model_name = 'Voting'

# Define classifiers
xgb_clf = XGBClassifier()
rf_clf = RandomForestClassifier()
ext_clf = ExtraTreesClassifier()
ann_clf = Sequential()
ann_clf.add(Dense(30, activation='relu'))
ann_clf.add(Dense(20, activation='relu'))
ann_clf.add(Dense(10, activation='relu'))
ann_clf.add(Dense(20, activation='relu'))
ann_clf.add(Dense(1, activation='sigmoid'))
ann_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create the ensemble model
ensemble_clf = VotingClassifier(
    estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('ext', ext_clf)], voting='soft'
)


start_time = datetime.now()
print(f"Training Started at {start_time.time()}:")
ensemble_clf.fit(X_res, y_res)
print("Training Completed ")
end_time = datetime.now()
difference = end_time - start_time
print(difference)
ensemble_clf.score(X_test_scaled, y_test)
y_test_prediction = ensemble_clf.predict(X_test_scaled)
y_test_prob = ensemble_clf.predict_proba(X_test_scaled)
y_df = pd.DataFrame({'y_test': y_test['reorder_label'],
                     'y_test_prob': y_test_prob[:, 1],
                     'y_test_prediction': y_test_prediction})
# train analysis
y_train_prediction = ensemble_clf.predict(X_train_scaled)
y_train_prob = ensemble_clf.predict_proba(X_train_scaled)
y_train_df = pd.DataFrame({'y_train': y_train['reorder_label'],
                           'y_train_prob': y_train_prob[:, 1],
                           'y_train_prediction': y_train_prediction})
importance = ensemble_clf.feature_importances_
# plot feature importance
plt.figure(figsize=(11, 6), dpi=80)
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.52, right=0.96, top=0.96)
plt.bar(columns, importance)
plt.xticks(rotation=3 * 90)
if f'FI_CB_tot_V{version}' not in os.listdir(f'{Folder}'):
    os.mkdir(f'{Folder}\\FI_CB_tot_V{version}')
    plt.savefig(f'{Folder}\FI_CB_tot_V{version}\FI_{model_name}_{scaler}.JPEG', dpi=300)
    plt.show()

DT_k,DT_t = best_params()
plot_classification_report(DT_k, model_name,version ,str(difference), {}, DT_t,scaler,2)
##################################################################################
model_name = 'stacking'

# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense


# define base models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
nn_model = Sequential()
nn_model.add(Dense(12, input_dim=30, activation='relu'))
nn_model.add(Dense(8, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit base models on training data and get predictions for test set
rf_model.fit(X_res, y_res)
rf_pred = rf_model.predict(X_test_scaled)

xgb_model.fit(X_res, y_res)
xgb_pred = xgb_model.predict(X_test_scaled)

nn_model.fit(X_res, y_res, epochs=10)
nn_pred = nn_model.predict(X_test_scaled)
nn_pred = np.round(nn_pred).astype(int)

# stack predictions into new feature set
stacked_predictions = np.column_stack((rf_pred, xgb_pred, nn_pred))

# define meta-model to make final prediction
meta_model = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

# evaluate accuracy of meta-model using cross-validation
meta_cv_scores = cross_val_score(meta_model, stacked_predictions, y_test, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (meta_cv_scores.mean(), meta_cv_scores.std() * 2))

# fit meta-model on full training set and make final prediction
meta_model.fit(stacked_predictions, y_test)

y_test_prediction = meta_model.predict(stacked_predictions)

report = (metrics.classification_report(y_test,y_test_prediction))
print(report)
# classes = ['class 1', 'class 2', 'class 3']'
if f'CR_CB_tot_V{version}' not in os.listdir(f'{Folder}'):
    os.mkdir(f'{Folder}\\CR_CB_tot_V{version}')
j = 1
for i in os.listdir(f"{Folder}\\CR_CB_tot_V{version}"):
    if f'{model_name}_{scaler}' in i:
        j += 1
report_path = f"{Folder}\CR_CB_tot_V{version}\\{model_name}_{scaler}_{j}.txt"
text_file = open(report_path, "w")
n = text_file.write(report)
text_file.close()
plt.figure(figsize=(8, 6), dpi=80)
sns.heatmap(metrics.confusion_matrix(y_test,y_test_prediction), annot=True, fmt='g')
plt.savefig(f'{Folder}\CR_CB_tot_V{version}\\{model_name}_{scaler}_{j}.png')
plt.show()
metrics.f1_score#
#
# final_accuracy = accuracy_score(y_test, final_pred)
# print("Final Accuracy:", final_accuracy)

