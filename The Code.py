
'''
important ###################
install this  to run our code
pip install plotly
pip install sklearn-som
'''
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import plotly.express as px
from pandas._libs.lib import tuples_to_object_array
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn_som.som import SOM
from numpy.core.multiarray import result_type
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.feature_selection import SequentialFeatureSelector


"""# part 1"""
#change to the path CSV in your OS
df=pd.read_csv('MCSDatasetNEXTCONLab.csv')

df.head()

df.info()

train_data = df[df['Day'].isin([0, 1, 2])]
test_data = df[df['Day'] == 3]

train_data

test_data

# Draw TSNE Funnction


def dispaly_TSNE(data_tsne_after_fit,y_for_color,title):
  fig = px.scatter(x=data_tsne_after_fit[:, 0], y=data_tsne_after_fit[:, 1], color=y_for_color)
  fig.update_layout(
      title=title,
      xaxis_title="First t-SNE",
      yaxis_title="Second t-SNE",
      width=800, height=600
  )
  fig.show()

"""## A"""

x_train=train_data.drop(["ID",'Day','Ligitimacy'],axis=1)
y_train=train_data['Ligitimacy']
x_test=test_data.drop(["ID",'Day','Ligitimacy'],axis=1)
y_test=test_data['Ligitimacy']

"""## B"""

GNB=GaussianNB()
GNB.fit(x_train,y_train)
GNB_pred=GNB.predict(x_test)
GNB_Confusion_matrix=metrics.confusion_matrix(y_test,GNB_pred)
GNB_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = GNB_Confusion_matrix, display_labels = [False, True])
GNB_cm_display.plot()
plt.title("GaussianNB")
plt.show()

GNB_f1_score=f1_score(y_test,GNB_pred)
print("F1 Score: ",GNB_f1_score)

KNN=KNeighborsClassifier()
KNN.fit(x_train,y_train)
KNN_pred=KNN.predict(x_test)
KNN_Confusion_matrix=metrics.confusion_matrix(y_test,KNN_pred)
KNN_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = KNN_Confusion_matrix, display_labels = [False, True])
KNN_cm_display.plot()
plt.title("KNeighborsClassifier")
plt.show()
KNN_f1_score=f1_score(y_test,KNN_pred)
print("F1 Score: ",KNN_f1_score)

"""## C"""

tsne = TSNE(n_components=2, random_state=0)
tsne_train = tsne.fit_transform(x_train)
dispaly_TSNE(data_tsne_after_fit=tsne_train,y_for_color=y_train,title="TSNE Training Set Plot")



tsne_test = tsne.fit_transform(x_test)
dispaly_TSNE(data_tsne_after_fit=tsne_test,y_for_color=y_test,title="TSNE Test Set Plot")

print('Shape of the Training Data befor T-SNE:', x_train.shape)
print('Shape of the Training Data after T-SNE:', tsne_train.shape)
print('Shape of the Testing Data befor T-SNE:', x_test.shape)
print('Shape of the Testing Data after T-SNE:', tsne_test.shape)

"""##**Part2**

## A
"""

# Define the range of dimensions/components for PCA and AE
dimensions = list(range(2, 11))

# Initialize lists to store F1 scores for each dimensionality reduction method
pca_f1_scores_nb = []
pca_f1_scores_knn = []
# Perform dimensionality reduction using PCA and AE for different dimensions
for n in dimensions:
    # PCA
    pca = PCA(n_components=n, random_state=0)
    pca_pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('pca', pca),
    ])
    X_train_pca = pca_pipeline.fit_transform(x_train)
    X_test_pca = pca_pipeline.transform(x_test)

    # Apply classifiers on the reduced data (after pca)
    GNB.fit(X_train_pca, y_train)
    GNB_pred_pca = GNB.predict(X_test_pca)
    GNB_f1_score_pca = f1_score(y_test, GNB_pred_pca)
    pca_f1_scores_nb.append(GNB_f1_score_pca)

    KNN.fit(X_train_pca, y_train)
    KNN_pred_pca = KNN.predict(X_test_pca)
    KNN_f1_score_pca = f1_score(y_test, KNN_pred_pca)
    pca_f1_scores_knn.append(KNN_f1_score_pca)

plt.plot(dimensions, pca_f1_scores_nb , marker='o', linestyle='-', color='b',label='(NB)')
plt.axhline(GNB_f1_score, linestyle='--', color='r', label='Baseline (NB)')
plt.plot(dimensions, pca_f1_scores_knn, marker='o', linestyle='-', color='g',label='(KNN)')
plt.axhline(KNN_f1_score, linestyle='--', color='y', label='Baseline (KNN)')
plt.title('PCA - F1 Score vs. Number of Components (NB)')
plt.xlabel('Number of Components')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

ae_f1_scores_nb = []
ae_f1_scores_knn = []
#dimension = list(range(x_train.shape[1], 1, -1))
dimension = list(range(2,11))
scaler= MinMaxScaler()
X_train = scaler.fit_transform(x_train,y_train)
X_test = scaler.transform(x_test)
for n in dimension:

    # inital autoencoder

    autoencoder = MLPRegressor(hidden_layer_sizes=[n], activation='relu', solver='adam',random_state=0)
    autoencoder.fit(X_train, X_train)

    # Reduce dimensionality with the trained autoencoder

    X_train_ae = autoencoder.predict(X_train)
    X_test_ae = autoencoder.predict(X_test)

    # Applying  the  classifiers on the reduced data (after ae)
    GNB.fit(X_train_ae, y_train)
    GNB_pred_ae = GNB.predict(X_test_ae)
    GNB_f1_score_ae = f1_score(y_test, GNB_pred_ae)
    ae_f1_scores_nb.append(GNB_f1_score_ae)

    KNN.fit(X_train_ae, y_train)
    KNN_pred_ae = KNN.predict(X_test_ae)
    KNN_f1_score_ae = f1_score(y_test, KNN_pred_ae)
    ae_f1_scores_knn.append(KNN_f1_score_ae)

plt.plot( dimension,ae_f1_scores_nb, marker='o', linestyle='-', color='b', label='AE_NB')
plt.axhline(GNB_f1_score, linestyle='--', color='r', label='Baseline (NB)')
plt.plot( dimension,ae_f1_scores_knn, marker='o', linestyle='-', color='g', label='AE_KNN')
plt.axhline(KNN_f1_score, linestyle='--', color='y', label='Baseline (KNN)')
plt.title('Autoencoder - F1 Score vs. Number of Components (NB)')
plt.xlabel('Number of Components')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

# Find the index of the best F1 score for PCA
best_pca_index_nb = pca_f1_scores_nb.index(max(pca_f1_scores_nb))
best_pca_index_knn = pca_f1_scores_knn.index(max(pca_f1_scores_knn))

# Find the index of the best F1 score for AE
best_ae_index_nb = ae_f1_scores_nb.index(max(ae_f1_scores_nb))
best_ae_index_knn = ae_f1_scores_knn.index(max(ae_f1_scores_knn))

# Get the best dimensions for PCA
best_pca_dimension_nb = dimensions[best_pca_index_nb]
best_pca_dimension_knn = dimensions[best_pca_index_knn]

# Get the best dimensions for AE
best_ae_dimension_nb = dimensions[best_ae_index_nb]
best_ae_dimension_knn = dimensions[best_ae_index_knn]

# Perform dimensionality reduction using PCA and AE with the best dimensions
best_pca_nb = PCA(n_components=best_pca_dimension_nb, random_state=0)
best_pca_knn = PCA(n_components=best_pca_dimension_knn, random_state=0)


best_pca_pipeline_nb = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', best_pca_nb),
])

best_pca_pipeline_knn = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', best_pca_knn),
])
X_train_best_pca_nb = best_pca_pipeline_nb.fit_transform(x_train)
X_test_best_pca_nb = best_pca_pipeline_nb.transform(x_test)

X_train_best_pca_knn = best_pca_pipeline_knn.fit_transform(x_train)
X_test_best_pca_knn = best_pca_pipeline_knn.transform(x_test)

"""## B"""



# Generate TSNE plots for the best performance in the previous part in PCA
tsne_train_best_pca_nb = TSNE(n_components=2, random_state=0).fit_transform(X_train_best_pca_nb)
tsne_test_best_pca_nb = TSNE(n_components=2, random_state=0).fit_transform(X_test_best_pca_nb)
tsne_train_best_pca_knn = TSNE(n_components=2, random_state=0).fit_transform(X_train_best_pca_knn)
tsne_test_best_pca_knn = TSNE(n_components=2, random_state=0).fit_transform(X_test_best_pca_knn)

#### Plot the TSNE plots
######TSNE - Best PCA NB

dispaly_TSNE(data_tsne_after_fit=tsne_train_best_pca_nb,y_for_color=y_train,title="TSNE - Best PCA NB (Train)")
dispaly_TSNE(data_tsne_after_fit=tsne_test_best_pca_nb,y_for_color=y_test,title="TSNE - Best PCA NB (Test)")

######TSNE - Best PCA KNN

dispaly_TSNE(data_tsne_after_fit=tsne_train_best_pca_knn,y_for_color=y_train,title="TSNE - Best PCA KNN (Train)")
dispaly_TSNE(data_tsne_after_fit=tsne_test_best_pca_knn,y_for_color=y_test,title="TSNE - Best PCA KNN (Test)")

#auto encoder

best_ae_nb = MLPRegressor(hidden_layer_sizes=[best_ae_dimension_nb], activation='relu', solver='adam',random_state=0)
best_ae_knn = MLPRegressor(hidden_layer_sizes=[best_ae_dimension_knn],activation='relu', solver='adam',random_state=0)

scaler= MinMaxScaler()
X_train = scaler.fit_transform(x_train,y_train)
X_test = scaler.transform(x_test)

# Reduce dimensionality with the trained autoencoder
best_ae_nb.fit(X_train,X_train)
X_train_best_ae_nb = best_ae_nb.predict(X_train)
X_test_best_ae_nb = best_ae_nb.predict(X_test)

# X_test_best_ae_nb = best_ae_pipeline_nb['ae'].predict(x_test)



# X_train_best_ae_knn = best_ae_pipeline_knn['ae'].fit(x_train,y_train)
# X_test_best_ae_knn = best_ae_pipeline_knn['ae'].predict(x_test)
best_ae_knn.fit(X_train,X_train)
X_train_best_ae_knn = best_ae_knn.predict(X_train)
X_test_best_ae_knn = best_ae_knn.predict(X_test)

# Generate TSNE plots for the best performance in the previous part in AE

tsne_train_best_ae_nb = TSNE(n_components=2, random_state=0).fit_transform(X_train_best_ae_nb)
tsne_test_best_ae_nb = TSNE(n_components=2, random_state=0).fit_transform(X_test_best_ae_nb)
tsne_train_best_ae_knn = TSNE(n_components=2, random_state=0).fit_transform(X_train_best_ae_knn)
tsne_test_best_ae_knn = TSNE(n_components=2, random_state=0).fit_transform(X_test_best_ae_knn)

##TSNE plot NB
dispaly_TSNE(data_tsne_after_fit=tsne_train_best_ae_nb,y_for_color=y_train,title="TSNE - Best AE NB (Train)")
dispaly_TSNE(data_tsne_after_fit=tsne_test_best_ae_nb,y_for_color=y_test,title="TSNE - Best AE NB (Test)")


##TSNE plot KNN
dispaly_TSNE(data_tsne_after_fit=tsne_train_best_ae_knn,y_for_color=y_train,title="TSNE - Best AE KNN (Train)")
dispaly_TSNE(data_tsne_after_fit=tsne_test_best_ae_knn,y_for_color=y_test,title="TSNE - Best AE KNN (Test)")

"""# part 3

## A
"""



def select_feature(X_train, y_train, X_test, y_test, FSM, model):
  fs = FSM
  fs.fit(X_train, y_train)
  X_train_new = fs.transform(X_train)
  X_test_new = fs.transform(X_test)
  model.fit(X_train_new, y_train)
  y_pred = model.predict(X_test_new)
  f1score = f1_score(y_test, y_pred)*100

  return f1score

def display_selected_feature(X_train, y_train, X_test, y_test, FSM,title):
  fs = FSM
  fs.fit(X_train, y_train)
  X_train_new = fs.transform(X_train)
  X_test_new = fs.transform(X_test)

  tsne_traing = TSNE(n_components=2, random_state=0).fit_transform(X_train_new)
  dispaly_TSNE(data_tsne_after_fit=tsne_traing,y_for_color=y_train,title=str(title+' train _data'))

  tsne_test = TSNE(n_components=2, random_state=0).fit_transform(X_test_new)
  dispaly_TSNE(data_tsne_after_fit=tsne_test,y_for_color=y_test,title=str(title+' test _data'))

GB = GaussianNB()
KNN=KNeighborsClassifier()

f1score_dict_GB = {}
f1score_dict_KNN = {}
for nf in range(2,11):
  fsm =  SelectKBest(chi2, k=nf)
  f1score = select_feature(X_train, y_train, X_test, y_test, fsm, GB)
  f1score_dict_GB[nf] = f1score
  f1score = select_feature(X_train, y_train, X_test, y_test, fsm, KNN)
  f1score_dict_KNN[nf] = f1score

plt.plot(*zip(*sorted(f1score_dict_GB.items())),marker='o', linestyle='-', color='b', label='GB')
plt.plot(*zip(*sorted(f1score_dict_KNN.items())),marker='o', linestyle='-', color='g', label='KNN')
plt.axhline(KNN_f1_score*100, linestyle='--', color='r', label='Baseline (KNN)')
plt.axhline(GNB_f1_score*100, linestyle='--', color='y', label='Baseline (GNB)')
Title = "Feature Selection with Chi-square Method"
plt.title(Title, fontsize=16)
plt.xlabel("Number of Features", fontsize=16)
plt.ylabel("f1 score (%)", fontsize=16)
plt.legend()

print("Maximum f1 score GB:", max(f1score_dict_GB.values()))
print("Best number of features GB:", max(f1score_dict_GB, key=f1score_dict_GB.get))

print("Maximum f1 score KNN:", max(f1score_dict_KNN.values()))
print("Best number of features KNN:", max(f1score_dict_KNN, key=f1score_dict_KNN.get))

GB = GaussianNB()
KNN=KNeighborsClassifier()

f1score_dict_GB = {}
f1score_dict_KNN = {}
for nf in range(2,11):
  fsm =  SelectKBest(mutual_info_classif, k=nf)
  f1score = select_feature(X_train, y_train, X_test, y_test, fsm, GB)
  f1score_dict_GB[nf] = f1score
  f1score = select_feature(X_train, y_train, X_test, y_test, fsm, KNN)
  f1score_dict_KNN[nf] = f1score

plt.plot(*zip(*sorted(f1score_dict_GB.items())),marker='o', linestyle='-', color='b', label='GB')
plt.plot(*zip(*sorted(f1score_dict_KNN.items())),marker='o', linestyle='-', color='g', label='KNN')
plt.axhline(KNN_f1_score*100, linestyle='--', color='r', label='Baseline (KNN)')
plt.axhline(GNB_f1_score*100, linestyle='--', color='y', label='Baseline (GNB)')
Title = "Feature Selection with mutual_info_classif Method"
plt.title(Title, fontsize=16)
plt.xlabel("Number of Features", fontsize=16)
plt.ylabel("f1 score (%)", fontsize=16)
plt.legend()

print("Maximum f1 score GB:", max(f1score_dict_GB.values()))
print("Best number of features GB:", max(f1score_dict_GB, key=f1score_dict_GB.get))

print("Maximum f1 score KNN:", max(f1score_dict_KNN.values()))
print("Best number of features KNN:", max(f1score_dict_KNN, key=f1score_dict_KNN.get))

"""## B"""




GB = GaussianNB()
KNN=KNeighborsClassifier()

f1score_dict_GB = {}
f1score_dict_KNN = {}
for nf in range(2,11):
  #inital the RFE
  fsm =  RFE(estimator=DecisionTreeClassifier(), n_features_to_select=nf)
  f1score = select_feature(X_train, y_train, X_test, y_test, fsm, GB)
  f1score_dict_GB[nf] = f1score
  f1score = select_feature(X_train, y_train, X_test, y_test, fsm, KNN)
  f1score_dict_KNN[nf] = f1score

plt.plot(*zip(*sorted(f1score_dict_GB.items())),marker='o', linestyle='-', color='g', label='GB')
plt.plot(*zip(*sorted(f1score_dict_KNN.items())),marker='o', linestyle='-', color='b', label='KNN')
Title = "Feature Selection with mutual_info_classif Method"  ################################
plt.title(Title, fontsize=16)
plt.axhline(KNN_f1_score*100, linestyle='--', color='r', label='Baseline (KNN)')
plt.axhline(GNB_f1_score*100, linestyle='--', color='y', label='Baseline (GNB)')
plt.xlabel("Number of Features", fontsize=16)
plt.ylabel("f1 score (%)", fontsize=16)
plt.legend()

print("Maximum f1 score GB:", max(f1score_dict_GB.values()))
print("Best number of features GB:", max(f1score_dict_GB, key=f1score_dict_GB.get))

print("Maximum f1 score KNN:", max(f1score_dict_KNN.values()))
print("Best number of features KNN:", max(f1score_dict_KNN, key=f1score_dict_KNN.get))

"""## C"""

fsm =  RFE(estimator=DecisionTreeClassifier(), n_features_to_select=3)
display_selected_feature(X_train, y_train, X_test, y_test,fsm,"TSNE best wrapper")

"""## part 4"""

train_data=pd.DataFrame(train_data)
train_data

train_data_x=train_data.drop(["ID",'Day'],axis=1)
x_train=train_data.drop(["ID",'Day','Ligitimacy','Hour','Minute','RemainingTime','Resources','Coverage','OnPeakHours','GridNumber','Duration'],axis=1)

y_train=train_data['Ligitimacy']
x_test=test_data.drop(["ID",'Day','Ligitimacy','Hour','Minute','RemainingTime','Resources','Coverage','OnPeakHours','GridNumber','Duration'],axis=1)
x_test_x=test_data.drop(["ID",'Day'],axis=1)

y_test=test_data['Ligitimacy']

def calculter(model_after_fit,x_test,data_test):
   culter={}
   for index, row in x_test.iterrows():
      elemat=pd.DataFrame(np.array(row).reshape(1, -1),columns=['Latitude','Longitude'])
      number_culster=model_after_fit.predict(elemat)

      Ligitimacy=((data_test.loc[(data_test['Longitude'] == row['Longitude']	) & (data_test['Latitude'] == row['Latitude'])])["Ligitimacy"]).astype(int)

      number_culster=int(number_culster)
      try:

       x=int(Ligitimacy)

      except:
        # to  remove duplicate and get on of them
         x=list(Ligitimacy)[0]


      try:
        cout=culter[number_culster]
        if(x==1 and cout!=-1):
           # for calculate the number ligamtate in each class
          culter[number_culster]=cout+1
        else:
          # for  fake classes not included
          culter[number_culster]=-1
      except:

       culter[number_culster]=x
   return culter

def culterdict(dim,data_dict):
  cout=0
  for i in range(dim):
    if(data_dict[i]!=-1):
      cout=cout+data_dict[i]
  return cout

"""## A"""


result={}
for i  in [8,12,16,20,32]:
    kmeans = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(x_train)
    #print("centroid",kmeans.cluster_centers_)
    x=calculter(kmeans,x_train,train_data)
    #print(x)
    score=culterdict(i,x)
    result[i]=score

result

myList = result.items()
print(myList)
myList = sorted(myList)
x, y = zip(*myList)
plt.xlabel('Number of custer')
plt.ylabel('Legitimate only clusters ')
plt.title("K-MEAS algorithm")
plt.plot(x, y,marker='o', linestyle='-', color='g')
plt.xlim([1, 35])
plt.legend()
plt.show()

"""## B"""


result={}
data_test=train_data
for i  in [8,12,16,20,32]:
    sofm=SOM(m=int(i/2), n=2, dim=2,random_state=0)
    sofm.fit(np.array(x_train))
    culter={}

    for index, row in x_train.iterrows():
        #elemat=pd.DataFrame(np.array(row).reshape(1, -1),columns=['Latitude','Longitude'])
        number_culster=sofm.predict(np.array(row).reshape(1, -1))
        #number_culster=model_after_fit.fit_predict(elemat)
        Ligitimacy=((data_test.loc[(data_test['Longitude'] == row[1]	) & (data_test['Latitude'] == row[0])])["Ligitimacy"]).astype(int)
        number_culster=int(number_culster)
        try:
         x=int(Ligitimacy)
        except:
          #for remove duplacates return form data set
          x=list(Ligitimacy)[0]
        try:
          cout=culter[number_culster]
          if(x==1 and cout!=-1):
            culter[number_culster]=cout+1
          else:
            culter[number_culster]=-1
        except:
          #for intiztion culster in dict
         culter[number_culster]=x
    cout=0
    for K in culter.keys():
      if(culter[K]!=-1):
        cout=cout+culter[K]

    result[i]=cout

myList = result.items()
print(myList)
myList = sorted(myList)
x, y = zip(*myList)
plt.xlabel('Number of custer')
plt.ylabel('Legitimate only clusters ')
plt.title("SOFM algorithm")
plt.plot(x, y,marker='o', linestyle='-', color='g')
plt.xlim([1, 35])
plt.legend()
plt.show()

"""## C"""

def calculter_DBSCAN(culters_number,x_test,data_test):
   culter={}
   counter=0
   for index, row in x_test.iterrows():

      number_culster=culters_number[counter]
      counter=counter+1

      Ligitimacy=((data_test.loc[(data_test['Longitude'] == row['Longitude']	) & (data_test['Latitude'] == row['Latitude'])])["Ligitimacy"]).astype(int)

      number_culster=int(number_culster)
      try:

       x=int(Ligitimacy)

      except:
         #print(list(Ligitimacy))
         x=list(Ligitimacy)[0]

      try:
        cout=culter[number_culster]
        if(x==1 and cout!=-1):

          culter[number_culster]=cout+1
        else:
          culter[number_culster]=-1
      except:
        #for intiztion culster in dict
       culter[number_culster]=x
   return culter



result={}
minPts = 5
#[0.01, .0001, .5, .00310, .0020]
for index ,i  in enumerate([0.001, .005, .002, .0001, .00632]):
    #kmeans = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(x_train)
    clustering = DBSCAN(eps=i, min_samples=7)
    #clustering.fit(np.array(x_train))
    number_culster=clustering.fit_predict(x_train)
    print("centroid",clustering.labels_)
    culter=calculter_DBSCAN(number_culster,x_train,train_data)
    print(x)
    cout=0
    print(culter)
    print(len(culter))
    for K in culter.keys():
      if(culter[K]!=-1):
        cout=cout+culter[K]

    result[i]=cout

result

myList = result.items()
print(myList)
myList = sorted(myList)
x, y = zip(*myList)
plt.xlabel('eplilon')
plt.ylabel('Legitimate only clusters ')
plt.title("DBSCAN algorithm")
plt.plot(x, y,marker='o', linestyle='-', color='g')
#plt.xlim(result.keys())
plt.legend()
plt.show()