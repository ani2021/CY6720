import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict, train_test_split, KFold

def load_dataset():
  ds = pd.read_csv('UNSW-NB15_2.csv').append(pd.read_csv('UNSW-NB15_3.csv')).append(pd.read_csv('UNSW-NB15_4.csv')).append(pd.read_csv('UNSW-NB15_1.csv'))
  return ds

def show_heat_map(corr):  
  sns.heatmap(corr)
  plt.show()

def feature_engineering(x, y, s, columns):
  for c in range(0, len(x[0])):
    ols = sm.OLS(y, x).fit()
    maxv = max(ols.pvalues).astype(float)
    if maxv > s:
      for q in range(0, len(x[0])-c):
        if (ols.pvalues[q].astype(float) == maxv):
          x = np.delete(x, q, 1)
  print(ols.summary())
  return x

def transform_features(labeled_ds):
  features = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
  'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'Sjit', 'Djit', 'Stime', 'Ltime', 
  'Sintpkt', 'Dintpkt', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'Label']
  for feature in features:
    labeled_ds[feature] = LabelEncoder().fit_transform(labeled_ds[feature]).astype('float64')
  return labeled_ds

# The code here takes the preprocessor and pipelines it to the LinearSVC model
# and then fits trains the model
def create_linear_svc(preprocessor, x_train, y_train):
  model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LinearSVC(penalty='l1', dual=False, class_weight='balanced'))])
  model.fit(x_train, y_train)
  return model

# The code here takes the preprocessor and pipelines it to the NearestCentroid model
# and then fits trains the model
def create_nearest_centroid(preprocessor, x_train, y_train):
  model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', NearestCentroid())])
  model.fit(x_train, y_train)
  return model

# The code below trains the models based on the x and y split
# model_score is tehn calculated based on the training data
# Once trained we calculate the precision score, accuracy score,
# mean squared error, cross score, and the confusion matrix
def calc_model_scores(model, x, y):
  model_score = model.score(x, y)
  model_prediction = model.predict(x)
  kf = KFold(n_splits=20, shuffle=True)
  precision_score = metrics.precision_score(y, model_prediction, average='micro')
  accuracy_score = metrics.accuracy_score(y, model_prediction)
  mean_squared_error = metrics.mean_squared_error(y, model_prediction)
  cross_score = cross_val_score(model, x, y, cv=kf)
  confustion_matrix = [] # metrics.confusion_matrix(x, model_prediction)
  return model_score, precision_score, accuracy_score, cross_score, confustion_matrix, mean_squared_error

# This prints out all the model analytics
def print_model_data(model_name, model_score, precision_score, accuracy_score, cross_score, confustion_matrix, mean_squared_error):
  print('\n' + model_name + ' Analysis\n')
  print(model_name + ' Cross Validation Score: ' + str(cross_score))
  print(model_name + ' Precision Validation Score: ' + str(precision_score))
  print(model_name + ' Accuracy Validation Score: ' + str(accuracy_score))
  print(model_name + ' Mean Squared Error: ' + str(mean_squared_error))
  print(model_name + ' Confusion Matrix: ' + str(confustion_matrix))
  print(model_name + ' Model Score: ' + str(model_score))
  print('\n')

def train_models(x, y):
  linearsvc = LinearSVC(penalty='l1', dual=False, class_weight='balanced')
  nearestcentroid = NearestCentroid()
  knn = KNeighborsClassifier(n_neighbors=33, weights='distance')
  models = [[nearestcentroid, 'NearestCentroid'], [knn, 'KNN'], [linearsvc, 'LinearSVC']]
  for model in models:
    model[0].fit(x, y)
  return models

def semisupervised_results(dsv1):
  # drop_features = ['trans_depth', 'res_bdy_len', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd']
  # dsv1.drop(drop_features, axis=1)
  unlabeled_x = dsv1.copy()[10000:60000]
  dsv1.replace('', np.nan, inplace=True)
  dsv1.dropna(subset=['attack_cat'], inplace=True)
  labeled_ds = dsv1[:10000]
  labeled_ds.iloc[:,-2] = LabelEncoder().fit_transform(labeled_ds.iloc[:,-2]).astype('float64')
  corr = labeled_ds.corr()
  show_heat_map(corr)
  y = LabelEncoder().fit_transform(labeled_ds['attack_cat'])
  unlabeled_x = transform_features(unlabeled_x.drop(['attack_cat', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_ftp_login'], axis=1))
  labeled_x = transform_features(labeled_ds.drop(['attack_cat', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_ftp_login'], axis=1))
  
  labeled_x = pd.DataFrame(feature_engineering(labeled_x.dropna().to_numpy(), y, 0.05, labeled_ds.values))

  x_train, x_test, y_train, y_test = train_test_split(labeled_x, y, test_size=0.2, random_state=100, shuffle=True)
  models = train_models(x_train, y_train)

  for model in models:
    model_score, precision_score, accuracy_score, cross_score, confustion_matrix, mean_squared_error = calc_model_scores(model[0], x_test, y_test)
    print_model_data(model[1], model_score, precision_score, accuracy_score, cross_score, confustion_matrix, mean_squared_error)

  unlabeled_x = pd.DataFrame(feature_engineering(unlabeled_x.dropna().to_numpy(), y, 0.05, unlabeled_x.values))
  for model in models:
    y = model[0].predict(unlabeled_x)
    model[0].fit(unlabeled_x, y)
    model_score, precision_score, accuracy_score, cross_score, confustion_matrix, mean_squared_error = calc_model_scores(model[0], unlabeled_x, y)
    print_model_data(model[1], model_score, precision_score, accuracy_score, cross_score, confustion_matrix, mean_squared_error)

def main():
  ds = load_dataset()
  semisupervised_results(ds.copy())

main()