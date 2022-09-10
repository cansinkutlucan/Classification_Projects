import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix, average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

df_male = pd.read_csv('https://query.data.world/s/h3pbhckz5ck4rc7qmt2wlknlnn7esr', encoding='latin-1')
df_female= pd.read_csv('https://query.data.world/s/sq27zz4hawg32yfxksqwijxmpwmynq')
df_male.head()
df_female.head()
df_male.shape
df_female.shape

data = pd.concat([df_male,df_female], axis=0, ignore_index=True)
data.head()
data.shape
data.info(verbose=True)

data.isnull().sum()

drop_list =[]
for col in data:
  if data[col].isnull().sum()>1800:
    print(f"{col} = {data[col].isnull().sum()}")
    drop_list.append(col)

data.drop(drop_list, axis=1, inplace=True)
data.isnull().sum()

data.DODRace.unique()
data["DODRace"] = data.DODRace.map({1: "White",2: "Black",3: "Hispanic",4: "Asian",5: "Native American",6: "Pacific Islander", 8: "Other"})

data.groupby(["Component"])["DODRace"].value_counts().plot(kind="barh")
data.groupby(["Component", "Branch"])["DODRace"].value_counts().plot(kind="barh") #.iplot(kind="barh")

drop_list1 = ["Date", "Installation", "Component", "Branch", "PrimaryMOS","Weightlbs", "Heightin"]
data.drop(drop_list1, axis=1, inplace=True)
data.columns
data.SubjectNumericRace.value_counts()
data.drop("SubjectNumericRace",axis=1,inplace=True)

data = data[(data["DODRace"] == "White") | (data["DODRace"] == "Black") | (data["DODRace"] == "Hispanic")]
data

data.reset_index(drop=True, inplace=True)

plt.figure(figsize=(20,20))
sns.heatmap(data.corr(), cmap ="coolwarm")

def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

########### DATA Preprocessing
# Train-Test-Split
X = data.drop("DODRace",axis=1)
y = data.DODRace
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify = y)

# One-hot encoder
cat = X.select_dtypes("object").columns
cat 
column_trans = make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse=False), cat), remainder=MinMaxScaler())

# IMPLEMENTATION ML MODELS
################ Logistic model
from sklearn.pipeline import Pipeline

# Modelling
operations = [("OneHotEncoder", column_trans), ("log", LogisticRegression(class_weight='balanced',max_iter=10000))]
pipe_log_model = Pipeline(steps=operations)
pipe_log_model.fit(X_train,y_train)
eval_metric(pipe_log_model, X_train, y_train, X_test, y_test)

# Cross Validation
scoring = {"f1_Hispanic" : make_scorer(f1_score, average = None, labels =["Hispanic"]),
           "precision_Hispanic" : make_scorer(precision_score, average = None, labels =["Hispanic"]),
           "recall_Hispanic" : make_scorer(recall_score, average = None, labels =["Hispanic"])}

operations = [("OneHotEncoder", column_trans), ("log", LogisticRegression(class_weight='balanced',max_iter=10000))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]  # f1_Hispanic:0.59, precision_Hispanic: 0.49, recall_Hispanic : 0.773

# Gridsearch
recall_Hispanic =  make_scorer(recall_score, average=None, labels=["Hispanic"])
param_grid = {"log__C": [1, 5, 10],'log__penalty': ["l1", "l2"],'log__solver': ['liblinear', 'lbfgs'],}
operations = [("OneHotEncoder", column_trans), ("log", LogisticRegression(class_weight='balanced',max_iter=10000))]
model = Pipeline(steps=operations)
log_model_grid = GridSearchCV(model,param_grid,verbose=3,scoring=recall_Hispanic,n_jobs=-1, cv=5)
log_model_grid.fit(X_train,y_train)
log_model_grid.best_estimator_
log_model_grid.best_params_
log_model_grid.best_score_

eval_metric(log_model_grid, X_train, y_train, X_test, y_test)

# Cross Validation
operations = [("OneHotEncoder", column_trans), ("log", LogisticRegression(C=1, class_weight='balanced',max_iter=10000,solver="liblinear", penalty="l2"))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring = scoring, cv = 5)
df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores.mean()[2:]   # f1_Hispanic: 0.653, test_precision_Hispanic: 0.671, test_recall_Hispanic: 0.635

# Precision-Recall Curve
from yellowbrick.classifier import PrecisionRecallCurve
operations = [("OneHotEncoder", column_trans), ("log", LogisticRegression(class_weight='balanced',max_iter=10000,solver="liblinear", penalty="l2"))]
model = Pipeline(steps=operations)
viz = PrecisionRecallCurve(model,per_class=True, classes= ["Black", "Hispanic", "White"],cmap="Set1")
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show();
viz.score_    #  ["Hispanic"]  0.7312984133598117

y_pred = log_model_grid.predict(X_test)
log_AP = viz.score_["Hispanic"]
log_f1 = f1_score(y_test, y_pred, average=None, labels=["Hispanic"])
log_recall = recall_score(y_test, y_pred, average=None, labels=["Hispanic"])

################ SVC
# Modelling
operations_svc = [("OneHotEncoder", column_trans), ("svc", SVC(class_weight="balanced"))]
pipe_svc_model = Pipeline(steps=operations_svc)
pipe_svc_model.fit(X_train, y_train)

eval_metric(pipe_svc_model, X_train, y_train, X_test, y_test)

# Cross Validation
model = Pipeline(steps=operations_svc)
scores = cross_validate(model, X_train, y_train, scoring = scoring, cv = 5)
df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores.mean()[2:] #  test_f1_Hispanic : 0.568 , test_precision_Hispanic: 0.453, test_recall_Hispanic: 0.762

# Gridsearch
param_grid = {'svc__C': [1,2],'svc__gamma': ["scale", "auto", 1, 0.1, 0.01]}
operations_svc = [("OneHotEncoder", column_trans), ("svc", SVC(class_weight="balanced",random_state=101))]
model = Pipeline(steps=operations_svc)
svm_model_grid = GridSearchCV(model,param_grid,verbose=3,scoring=recall_Hispanic,n_jobs=-1)
svm_model_grid.fit(X_train, y_train)
svm_model_grid.best_params_
svm_model_grid.best_score_

eval_metric(svm_model_grid, X_train, y_train, X_test, y_test)

# Cross Validation
operations = [("OneHotEncoder", column_trans), ("svc", SVC(C=2, class_weight="balanced",gamma = 'scale'))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring = scoring, cv = 5)
df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores.mean()[2:] # test_f1_Hispanic: 0.582, test_precision_Hispanic: 0.470, test_recall_Hispanic: 0.764

# Precision-Recall Curve
operations_svc = [("OneHotEncoder", column_trans), ("svc", SVC(C=2, class_weight="balanced",gamma = 'scale'))]
model = Pipeline(steps=operations_svc)
viz = PrecisionRecallCurve( model,per_class=True, classes= ["Black", "Hispanic", "White"],cmap="Set1")
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show();
viz.score_  # 'Hispanic': 0.7182860644867964

y_pred = svm_model_grid.predict(X_test)
svc_AP = viz.score_["Hispanic"]
svc_f1 = f1_score(y_test, y_pred, average=None, labels=["Hispanic"])
svc_recall = recall_score(y_test, y_pred, average=None, labels=["Hispanic"])

################ RF
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
column_trans = make_column_transformer((ord_enc, cat), remainder='passthrough')

# Modelling
from sklearn.ensemble import RandomForestClassifier
operations_rf = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestClassifier(class_weight="balanced"))]
pipe_model_rf = Pipeline(steps=operations_rf)
pipe_model_rf.fit(X_train, y_train)

eval_metric(pipe_model_rf, X_train, y_train, X_test, y_test)

# Cross Validation
operations_rf = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestClassifier(class_weight="balanced"))]
model = Pipeline(steps=operations_rf)
scores = cross_validate(model, X_train, y_train, scoring = scoring, cv = 5)
df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores.mean()[2:] # test_f1_Hispanic: 0.073, test_precision_Hispanic: 0.743, test_recall_Hispanic: 0.039

# Gridsearch
param_grid = {'RF_model__n_estimators':[60,80,100],
             'RF_model__max_depth':[1,2,3],
             'RF_model__max_features':[4,6,8 ,"auto"],
             'RF_model__min_samples_split':[20,22,24],
             'RF_model__min_samples_leaf': [16,18,22]} 
operations_rf = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestClassifier(class_weight="balanced"))]
model = Pipeline(steps=operations_rf)
rf_grid_model = GridSearchCV(model,param_grid,verbose=3,scoring=recall_Hispanic, n_jobs=-1)
rf_grid_model.fit(X_train, y_train)
rf_grid_model.best_params_
rf_grid_model.best_score_

eval_metric(rf_grid_model, X_train, y_train, X_test, y_test)

# Cross Validation
operations = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestClassifier(class_weight="balanced", max_depth=1, n_estimators=60,min_samples_leaf=18,max_features = 4,min_samples_split= 20))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring = scoring, cv = 5)
df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores.mean()[2:]  # test_f1_Hispanic          0.279

# Precision-Recall Curve
operations_rf = [("OrdinalEncoder", column_trans), ("RF_model", RandomForestClassifier(class_weight="balanced", max_depth=1, n_estimators=60,min_samples_leaf=18,max_features = 4,min_samples_split= 20))]
model = Pipeline(steps=operations_rf)
viz = PrecisionRecallCurve(model,per_class=True, classes= ["Black", "Hispanic", "White"],cmap="Set1")
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show();

y_pred = rf_grid_model.predict(X_test)
rf_AP = viz.score_["Hispanic"]
rf_f1 = f1_score(y_test, y_pred, average=None, labels=["Hispanic"])
rf_recall = recall_score(y_test, y_pred, average=None, labels=["Hispanic"])

################ XGBoost
#!pip install xgboost==0.90
import xgboost as xgb
xgb.__version__ # 0.90

# Modelling
operations_xgb = [("OrdinalEncoder", column_trans), ("XGB_model", XGBClassifier(random_state=101))]
pipe_model_xgb = Pipeline(steps=operations_xgb)
pipe_model_xgb.fit(X_train, y_train)

# Class weight adjustment
from sklearn.utils import class_weight
classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
classes_weights
# compute_sample_weight y_train deki sayılara göre katsayı/ağırlık belirleyecek.
# y_train deki her bir örnek için üretiyor

comp = pd.DataFrame(classes_weights)
comp["label"] = y_train.reset_index(drop=True)
comp.groupby("label")[0].value_counts() # 1.48, 2.83 , 0.50 ... Bunlar yeni class weightlerimiz. Hepsini orta noktada dengelemiş oldu

pipe_model_xgb.fit(X_train,y_train, XGB_model__sample_weight=classes_weights)
# weight parameter in XGBoost is per instance not per class. Therefore, we need to assign the weight of each class to its instances, which is the same thing.

eval_metric(pipe_model_xgb, X_train, y_train, X_test, y_test)

# Cross Validation
operations_xgb = [("OrdinalEncoder", column_trans), ("XGB_model", XGBClassifier())]
model = Pipeline(steps=operations_xgb)
scores = cross_validate(model, X_train, y_train, scoring = scoring, cv = 5,fit_params={"XGB_model__sample_weight":classes_weights})
df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores.mean()[2:]   # test_f1_Hispanic          0.528 , # test_precision_Hispanic   0.695 # test_recall_Hispanic      0.427

# Gridsearch
param_grid = {"XGB_model__n_estimators":[100,200,300,400],'XGB_model__max_depth':[1,2],"XGB_model__learning_rate": [0.1, 0.3],"XGB_model__subsample":[0.8, 1],"XGB_model__colsample_bytree":[0.8, 1]}
operations_xgb = [("OrdinalEncoder", column_trans), ("XGB_model", XGBClassifier())]
model = Pipeline(steps=operations_xgb)
xgb_grid_model = GridSearchCV(model, param_grid, scoring=recall_Hispanic, n_jobs = -1, verbose = 2).fit(X_train, y_train, XGB_model__sample_weight=classes_weights)
xgb_grid_model.best_estimator_
xgb_grid_model.best_params_
xgb_grid_model.best_score_

eval_metric(xgb_grid_model, X_train, y_train, X_test, y_test)

# Cross Validation
operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBClassifier(learning_rate=0.3, max_depth=1, subsample=0.8,colsample_bytree=0.8, n_estimators=300))]
model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring = scoring, cv = 5)
df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores.mean()[2:] # test_f1_Hispanic: 0.425 # test_precision_Hispanic   0.707, test_recall_Hispanic: 0.306

# Precision-Recall Curve
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, plot_roc_curve, roc_auc_score, roc_curve
perations_xgb = [("OrdinalEncoder", column_trans), ("XGB_model", XGBClassifier(learning_rate=0.3, max_depth=1, subsample=0.8,colsample_bytree=0.8, n_estimators=300))]
model = Pipeline(steps=operations_xgb)
model.fit(X_train, y_train, XGB_model__sample_weight=classes_weights)
y_pred_proba = model.predict_proba(X_test)
precision_recall_curve(y_test, y_pred_proba)
plt.show() # 0.72

y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
average_precision_score(y_test_dummies[:, 1], y_pred_proba[:, 1])

y_pred = xgb_grid_model.predict(X_test)
xgb_AP = average_precision_score(y_test_dummies[:, 1], y_pred_proba[:, 1])
xgb_f1 = f1_score(y_test, y_pred, average=None, labels=["Hispanic"])
xgb_recall = recall_score(y_test, y_pred, average=None, labels=["Hispanic"])

############# COMPARING MODELS
compare = pd.DataFrame({"Model": ["Logistic Regression", "SVM",  "Random Forest", "XGBoost"],
                        "F1": [log_f1[0], svc_f1[0], rf_f1[0], xgb_f1[0]],
                        "Recall": [log_recall[0], svc_recall[0], rf_recall[0], xgb_recall[0]],
                        "AP": [log_AP, svc_AP, rf_AP, xgb_AP]})

def labels(ax):
    for p in ax.patches:
        width = p.get_width()                        # get bar length
        ax.text(width,                               # set the text at 1 unit right of the bar
                p.get_y() + p.get_height() / 2,      # get Y coordinate + X coordinate / 2
                '{:1.3f}'.format(width),             # set variable to display, 2 decimals
                ha = 'left',                         # horizontal alignment
                va = 'center')                       # vertical alignment
plt.figure(figsize=(14,10))
plt.subplot(311)
compare = compare.sort_values(by="F1", ascending=False)
ax=sns.barplot(x="F1", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.subplot(312)
compare = compare.sort_values(by="Recall", ascending=False)
ax=sns.barplot(x="Recall", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.subplot(313)
compare = compare.sort_values(by="AP", ascending=False)
ax=sns.barplot(x="AP", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.show()

# I get best scores in logistic Regression

# I'll work on these methods afterwards to balance data's classes and get better scores
# SMOTE
# Logistic Regression Over/Under
# SHAP






