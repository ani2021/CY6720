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
  ds = pd.read_csv('Project Datasets\\UNSW-NB15_2.csv').append(pd.read_csv('Project Datasets\\UNSW-NB15_3.csv')).append(pd.read_csv('Project Datasets\\UNSW-NB15_4.csv')).append(pd.read_csv('Project Datasets\\UNSW-NB15_1.csv'))
  return ds

# def create_preprocessor():
#   # The get_preprocessor method is used to preprocess the dataset 
#   features = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
#   'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'Sjit', 'Djit', 'Stime', 'Ltime', 
#   'Sintpkt', 'Dintpkt', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'Label']
#   # The line below is used to replace missing values with the most frequent values of the dataset and then append
#   # the StandardScaler to it
#   cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
#   transformer = ColumnTransformer(transformers=[('cat', cat_transformer, features)])
#   return transformer

def create_preprocessor():
  # The get_preprocessor method is used to preprocess the dataset 
  features = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
  'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 
  'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 
  'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
  # The line below is used to replace missing values with the most frequent values of the dataset and then append
  # the StandardScaler to it
  cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
  transformer = ColumnTransformer(transformers=[('cat', cat_transformer, features)])
  return transformer

# The code here takes the preprocessor and pipelines it to the LinearSVC model
# and then fits trains the model
def create_linear_svc(preprocessor, x_train, y_train):
  model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LinearSVC(penalty='l1', dual=False, class_weight='balanced'))])
  model.fit(x_train, y_train)
  return model

# The code here takes the preprocessor and pipelines it to the NuSVC model
# and then fits trains the model
# def create_nu_svc(preprocessor, x_train, y_train):
#   model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', NuSVC(nu=0.9, kernel='linear'))])
#   model.fit(x_train, y_train)
#   return model

# The code here takes the preprocessor and pipelines it to the NearestCentroid model
# and then fits trains the model
def create_nearest_centroid(preprocessor, x_train, y_train):
  model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', NearestCentroid())])
  model.fit(x_train, y_train)
  return model

# The code here takes the preprocessor and pipelines it to the KNeighborsClassifier model
# and then fits trains the model
def create_knn(preprocessor, x_train, y_train):
  model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier(n_neighbors=33, weights='distance'))])
  model.fit(x_train, y_train)
  return model

def create_DBSCAN_model(preprocessor):
  # This method generates the DBSCAN model for our dataset
  # it appends the preprocesser passed to the method before applying the model
  dbs = DBSCAN(eps=0.9, min_samples=14, leaf_size=50)
  model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', dbs)])
  return model

def create_kmeans_model(preprocessor):
  models = []
  # This method generates the k-means model for our dataset
  # it appends the preprocesser passed to the method before applying the model.
  # Here
  kms = [KMeans(n_clusters=9)]
  # kms = [KMeans(n_clusters=2), KMeans(n_clusters=3), KMeans(n_clusters=4), KMeans(n_clusters=5), KMeans(n_clusters=6)]
  for km in kms:
    models.append(Pipeline(steps=[('preprocessor', preprocessor), ('classifier', km)]))
  return models

# The code below trains the models based on the x and y split
# model_score is tehn calculated based on the training data
# Once trained we calculate the precision score, accuracy score,
# mean squared error, cross score, and the confusion matrix
def train_data(model, x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, shuffle=True)
  model.fit(x_train, y_train)
  model_score = model.score(x_test, y_test)
  model_prediction = model.predict(x_test)
  kf = KFold(n_splits=20, shuffle=True)
  precision_score = metrics.precision_score(y_test, model_prediction, average='micro')
  accuracy_score = metrics.accuracy_score(y_test, model_prediction)
  mean_squared_error = metrics.mean_squared_error(y_test, model_prediction)
  cross_score = cross_val_score(model, x_test, y_test, cv=kf)
  confustion_matrix = metrics.confusion_matrix(y_test, model_prediction)
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

def display_graph(ds, clusters):
  # This method displays all the clusters of the dataset
  colors = ['purple', 'red', 'blue', 'black']
  # This loop goes from 0 to clusters which is the number of clusters in the datasets that are being
  # passed through
  for i in range(0, clusters):
    plt.scatter(ds[i]['attack_cat'], ds[i]['Label'], color=colors[i], marker=".", linewidth=0)
  plt.xlabel('attack_cat')
  plt.ylabel('Label')
  # Here we are setting the y limit to -80 and 80. This is to prevent outliers from affecting
  # the graph
  plt.ylim(-80, 80)
  # This line is to set the title of the graph
  plt.title("AWS Honeypot Geo")
  plt.show()

def display_see_graph(rng, see):
  # This method is used to display the optimal value of k through the use
  # of the elbow method
  plt.title("Optimal K")
  plt.xlabel("K Value")
  plt.ylabel("Sum of Squared Error")
  # THis line is to plot the data for the graph
  plt.plot(rng, see)
  plt.show()

def display_sum_square_error_graph(rng, scores):
  # This method is used to display the optimal value of k but instead of
  # using the elbow method it is using the silhouette score which are stored
  # in the scores variable
  plt.plot(rng, scores,"bo-", color='orange')
  plt.xlabel("K")
  plt.ylabel("silhouette score")
  plt.grid(True)
  k_max = np.argmax(scores) + 2
  plt.axvline(x=k_max, label=['Optimal K'].format(k_max))
  plt.scatter(k_max, scores[k_max-2])
  plt.legend(shadow=True)
  plt.show()

def feature_engineering(x, y, s):
  for c in range(0, len(x[0])):
    ols = sm.OLS(y, x).fit()
    maxv = max(ols.pvalues).astype(float)
    if maxv > s:
      for q in range(0, len(x[0])-c):
        if (ols.pvalues[q].astype(float) == maxv):
          x = np.delete(x, q, 1)
  print(ols.summary())
  return x

def semisupervised_results(dsv1):
  # drop_features = ['trans_depth', 'res_bdy_len', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd']
  # dsv1.drop(drop_features, axis=1)
  unlabeled_x = dsv1.copy().drop(['attack_cat'], axis=1)[10000:60000]
  dsv1['attack_cat'].replace('', np.nan, inplace=True)
  dsv1.dropna(subset=['attack_cat'], inplace=True)
  labeled_ds = dsv1[:10000]
  print(labeled_ds.head())
  # This is all the y from the labeled dataset
  y = LabelEncoder().fit_transform(labeled_ds['attack_cat'])
  labeled_x = labeled_ds.drop(['attack_cat'], axis=1)
  preprocessor = create_preprocessor()
  labeled_x_2 = preprocessor.fit_transform(labeled_x)
  labeled_x_2 = feature_engineering(labeled_x_2.dropna().to_numpy(), y, 0.6)


  x_train, x_test, y_train, y_test = train_test_split(labeled_x, y, test_size=0.2, random_state=100, shuffle=True)
  
  # The lines below trains the models for the challenge and stores them
  # into the models list
  linearsvc = create_linear_svc(preprocessor, x_train, y_train)
  knn = create_knn(preprocessor, x_train, y_train)
  nearestcentroid = create_nearest_centroid(preprocessor, x_train, y_train)
  models = [[nearestcentroid, 'NearestCentroid', create_nearest_centroid], [knn, 'KNN', create_knn], [linearsvc, 'LinearSVC', create_linear_svc]]
  
  # This loops through all the models and prints the analytics of each supervised model
  # before combining it with the unlabeled data
  print('Before Merge\n')
  for model in models:
    model_score, precision_score, accuracy_score, cross_score, confustion_matrix, mean_squared_error = train_data(model[0], labeled_x, y)
    print_model_data(model[1], model_score, precision_score, accuracy_score, cross_score, confustion_matrix, mean_squared_error)
  
  print('After Merge\n')
  for model in models:
    model[2](preprocessor, x_train, y_train)
    y = model[0].predict(unlabeled_x)
    model_score, precision_score, accuracy_score, cross_score, confustion_matrix, mean_squared_error = train_data(model[0], unlabeled_x, y)
    print_model_data(model[1], model_score, precision_score, accuracy_score, cross_score, confustion_matrix, mean_squared_error)

def execute_DBSCAN(dsv2):
  print('Inside unsupervised_results')
  preprocessor = create_preprocessor()
  # This is used to execute the dbscan algorithm
  dbscan_model = create_DBSCAN_model(preprocessor)
  print()
  print("DBS Model Output")
  dsv2 = dsv2.drop(['attack_cat'], axis=1)[:1000]
  dbs_prediction = dbscan_model.fit(dsv2)
  dsv2['dbsclusters'] = dbs_prediction
  finalds = []
   # The loop below stores the different clusters based
   # on dbs_prediction
  for p in dbs_prediction:
    finalds.append(dsv2[dsv2.dbsclusters == p])
  print("DBS Prediction: " + str(dbs_prediction))

def execute_kmeans(dsv2):
  preprocessor = create_preprocessor()
  # This method is used to execute the k-means algorithm
  kmeans_models = create_kmeans_model(preprocessor)
  # The value we ended up using here is using the optimal
  # k value. The create_kmeans_model method returns us 
  # an array of kmeans models from 2-6 and since 3 is 
  # the optimal value I have changed it below to use that
  km_prediction = kmeans_models[0].fit_predict(dsv2)
  print("K-Means Prediction: " + str(km_prediction))
  dsv2['kmclusters'] = km_prediction
  finalds = []
  # The loop below stores the different clusters based on 
  # km_prediction
  for p in km_prediction:
    finalds.append(dsv2[dsv2.kmclusters == p])
  # display_graph(finalds, 3)
  rng = range(2, 7)
  x_transformed = pd.DataFrame(preprocessor.fit_transform(dsv2))
  sum_square_error = []
  scores = []
  # The method below is used to generate the differnt cluster outcomes
  # based on the value of k shown below. We use this output to display 
  # the graphs based on the elbow method and silhouette score
  for i in rng:
    km = KMeans(n_clusters=i)
    km.fit(x_transformed)
    scores.append(silhouette_score(x_transformed, km.labels_))
    sum_square_error.append(km.inertia_)
  # The methods below display the optimal values
  # of k
  # display_see_graph(rng, sum_square_error)
  # display_sum_square_error_graph(rng, scores)

def main():
  ds = load_dataset()
  semisupervised_results(ds.copy())
  print("\nSuccessful Execution!")

main()