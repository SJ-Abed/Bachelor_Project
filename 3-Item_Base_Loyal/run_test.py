import os
os.listdir()
from prepare_to_run import prepare
from run_classical_models import run
# os.chdir('Bachelor Project')
from notify import notifier
'''import warnings
warnings.filterwarnings('ignore')'''
body = prepare.create_res('3_1',2,'maxmin')
notifier.mail(body)
body = prepare.create_res('3_2',2,'maxmin')
notifier.mail(body)
body = prepare.create_res('3_3',2,'maxmin')
notifier.mail(body)
body = prepare.create_res('3_4',2,'maxmin')
notifier.mail(body)
body = prepare.create_res('3_5',2,'maxmin')
notifier.mail(body)
body = prepare.create_res('3_6',2,'maxmin')
notifier.mail(body)
body = prepare.create_res('3_8',2,'maxmin')
notifier.mail(body)


X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns = prepare.get_data('3_8',2,scaler='maxmin')

# prepare.create_res(3,2)
X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns = prepare.get_data('2',2,scaler='maxmin')
########################################################################################################################
########################################################################################################################
import seaborn as sns
from sklearn import metrics
#
run.simple("LGR",3,X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,2)
run.simple("GNB",3,X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,2)
run.simple("ADA",3,X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,2)
run.simple("RF",3,X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,2)
run.simple("DT",3,X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,2)
run.simple("EXT",3,X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,2)
run.simple("XGB",3,X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,2)
#





X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns = prepare.get_data('3',2,scaler='maxmin')
run.simple("RF",'3',X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,'maxmin',2,params={'min_samples_split':20,'min_samples_leaf':10,'criterion': 'gini','n_estimators': 200})



body = prepare.create_res('3_1',2,'maxmin')
notifier.mail(body)
body = prepare.create_res('3_2',2,'maxmin')
notifier.mail(body)
body = prepare.create_res('3_3',2,'maxmin')
notifier.mail(body)
body = prepare.create_res('3_4',2,'maxmin')
notifier.mail(body)
body = prepare.create_res('3_5',2,'maxmin')
notifier.mail(body)
body = prepare.create_res('3_6',2,'maxmin')
notifier.mail(body)

X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns = prepare.get_data('3_1',2,scaler='maxmin')
run.simple("XGB",'3',X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,'maxmin',2)
del X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns
X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns = prepare.get_data('3_2',2,scaler='maxmin')
run.simple("XGB",'3_2',X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,'maxmin',2)
del X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns
X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns = prepare.get_data('3_3',2,scaler='maxmin')
run.simple("XGB",'3_3',X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,'maxmin',2)
del X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns
X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns = prepare.get_data('3_4',2,scaler='maxmin')
run.simple("XGB",'3_4',X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,'maxmin',2)
del X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns
X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns = prepare.get_data('3_5',2,scaler='maxmin')
run.simple("XGB",'3_5',X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,'maxmin',2)
del X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns
X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns = prepare.get_data('3_6',2,scaler='maxmin')
run.simple("XGB",'3_6',X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,'maxmin',2)
del X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns
notifier.mail('all of the XGBs has been learned')
#########################
for v in range(13,25):
    body = prepare.create_res(f'3_{v}', 2, 'maxmin')
    notifier.mail(body)

for v in range(13,25):
    X_res, y_res, X_test_scaled, y_test, X_train_scaled, y_train, columns = prepare.get_data(f'3_{v}', 2, scaler='maxmin')
    run.simple("RF",f'3_{v}',X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns,'maxmin',2)
    del X_res,y_res,X_test_scaled,y_test,X_train_scaled,y_train,columns
notifier.mail('all of the XGBs has been learned')
#########################
# params = {'C': [0.1,0.5,1, 10],'penalty':[ 'l2', 'elasticnet', None],'solver':['lbfgs','liblinear','newton-cg', 'newton-cholesky', 'sag', 'saga'],'max_iter':[200]}
# params = {'C': [0.1,1],'penalty':['l2','l1'],'solver':['lbfgs','sag', 'saga'],'max_iter':[200]}
# run.tuner("LGR",3,X_res,y_res,params,10,2)
#
# params = {'var_smoothing': [1e-9,1e-10,1e-8,1e-11]}
# run.tuner("GNB",3,X_res,y_res,params,10,2)
# params = {'n_estimators': [10,50,100,200],'criterion':['gini','entropy','log_loss'],'min_samples_split':[2,3,5,8],'max_features':['sqrt','log2',None],'min_samples_leaf':[1,100,50,1000,5000],'min_weight_fraction_leaf':[0.0,0.1,0.5,0.9]}
params = {'min_samples_split':[2],'criterion':['gini','log_loss'],'n_estimators': [10,50,200],'min_samples_leaf':[1],'max_depth':[30,60],'min_weight_fraction_leaf':[0.0,0.2,0.5]}
run.tuner("RF",3,X_res,y_res,params,10,2)

# params = {'n_estimators': [10,50,100,200],'criterion':['gini','entropy','log_loss'],'min_samples_split':[2,3,5,8],'max_features':['sqrt','log2',None],'min_samples_leaf':[1,100,50,1000,5000],'min_weight_fraction_leaf':[0.0,0.1,0.5,0.9]}
params = {'min_samples_split':[2],'criterion':['gini','entropy','log_loss'],'n_estimators': [10,50,200],'min_samples_leaf':[1],'max_depth':[30,60],'min_weight_fraction_leaf':[0.0,0.2,0.5]}
run.tuner("EXT",3,X_res,y_res,params,10,2)

params = {'n_estimators': [10,50,100,200],'learning_rate':[0.1,1,10],'algorithm':['SAMME','SAMME.R']}
run.tuner("ADA",3,X_res,y_res,params,10,2)

# params = param_grid = {'learning_rate': [0.1, 0.05, 0.01],'max_depth':[3, 4, 5],'n_estimators': [100, 500, 1000]}
params = param_grid = {'learning_rate': [0.1],'max_depth':[3,  5],'n_estimators': [100]}
run.tuner("XGB",3,X_res,y_res,params,10,2)
