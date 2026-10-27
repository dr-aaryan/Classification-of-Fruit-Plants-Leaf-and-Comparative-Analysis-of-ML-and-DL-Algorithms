# Classification-of-Fruit-Plants-Leaf-and-Comparative-Analysis-of-ML-and-DL-Algorithms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pickle
 
##
for dirname, _, filenames in os.walk('E:/FruitLeafDataset'):
    for filename in filenames:
        os.path.join(dirname, filename)

DATADIR = 'E:/FruitLeafDataset'

CATEGORIES = ['AnanasComosus',
              'AnnonaReticulata',
              'AnnonaSquamosa',
              'ArtocarpusHeterophyllus',
              'ArtocarpusLacucha',
              'CeltisSinensis',
              'CitrusAurantiifolia',
              'CitrusLimon',
              'CitrusMaxima',
              'CitrusPyriformis',
              'CordiaMyxa',
              'FicusRacemosa',
              'GrewiaAsiatica',
              'IpomoeaBatatas',
              'LimoniaAcidissima',
              'LitchiChinensis',
              'MadhucaLongifolia',
              'MangiferaIndica',
              'ManilkaraHexandra',
              'ManilkaraZapota',
              'MorusAlba',
              'NeolamarckiaCadamba',
              'PithecellobiumDulce'
              'PsidiumGuajava',
              'PunicaGranatum',
              'SolanumIycopersicum',
              'SolanumNigrum',
              'SyzygiumJambos',
              'TamarindusIndica',
              'ZizphusMauritiana']
             
IMG_SIZE=250

##
for category in CATEGORIES:
    path=os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        plt.imshow(img_array)
        plt.show()
        break
    break

##
training_data=[]
def create_training_data():
    for category in CATEGORIES:
        print(category)
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()
print(len(training_data))
lenofimage = len(training_data)

##
X=[]
y=[]

for categories, label in training_data:
    X.append(categories)
    y.append(label)
X= np.array(X).reshape(lenofimage,-1)
print(X.shape)
X = X/155.0
y=np.array(y)
y.shape

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

Classifier_name=['LogisticRegression',
                 'KNeighborsClassifier',
                 'SVC',
                 'DecisionTreeClassifier',
                 'MLPClassifier',
                 'GaussianNB']

classifiers = [
    LogisticRegression(multi_class='multinomial', random_state=1),
    KNeighborsClassifier(11),
    SVC(kernel="linear", C=0.045),
    DecisionTreeClassifier(max_depth=1000),
    MLPClassifier(alpha=1, max_iter=10000),
    GaussianNB()]

## Classification setting for whole features.....

itr=11
dim=(len(classifiers),5)
Eval_matrix=np.zeros(dim)

scoring=['fit_time','score_time','test_precision_macro', 'test_recall_macro']
#applying PCA to the dataset
print("X shape: ",X.shape)
Done_ML_Algo=[]
Results_clf_itr=[]
comp_Val=0
PCA_itr=12
for r in range(PCA_itr):
    print("PCA itr=",r, "/",PCA_itr)
    comp_Val+=10
    n_components=(comp_Val)
    pca=PCA()
    X_new=pca.fit_transform(X)
    Results=[]
    i=0
    for clf in classifiers:
      print("Algo.: ", Classifier_name[i])
      R1=0;R2=0;R3=0;R4=0;R5=0;R6=0;R7=0;R8=0;R9=0
      for j in range(itr):
        print('Iteration=',j,"Algo.: ", Classifier_name[i])
        X_train, X_test, y_train, y_test = train_test_split(X_new,y)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pp=classification_report(y_test,y_pred, output_dict=True)
        print("Accuracy on unknown data is",classification_report(y_test,y_pred))
        R1+=pp['accuracy']
        R2+=pp['macro avg']['precision']
        R3+=pp['macro avg']['recall']
        R4+=pp['macro avg']['f1-score']
        R5+=pp['macro avg']['support']
   
        R6+=pp['weighted avg']['precision']
        R7+=pp['weighted avg']['recall']
        R8+=pp['weighted avg']['f1-score']
        R9+=pp['weighted avg']['support']
      R=[R1/itr,R2/itr,R3/itr,R4/itr, R6/itr,R7/itr,R8/itr]
      print("Algo.: ", Classifier_name[i],"Accuracy:",  R[0])
     
      Results.append(R)
      Done_ML_Algo.append(Classifier_name[i])
      i+=1
      pickle.dump(Results, open("E:/FruitLeafDataset.dat", "wb"))
    Results_clf_itr.append(Results)
   
    for q in Results_clf_itr:
      print(q)
Matrix_results_itr_Feature=np.array(Results_clf_itr)
print("Done.... all iterations")

## Feature selection trend to find the subset of feature using PCA

import pickle
itr=14
scoring=['fit_time','score_time','test_precision_macro', 'test_recall_macro']
Results_clf_itr=[]
comp_Val=0
PCA_itr=15

pca=PCA()
X_pca=pca.fit_transform(X)

for r in range(PCA_itr):
    print("PCA itr=",r, "/",PCA_itr)
    comp_Val+=3
    n_components=(comp_Val)
    X_new=X_pca[:, :comp_Val]
    Results=[]
    i=0
    for clf in classifiers:
      print("Algo.: ", Classifier_name[i])
      R1=0
      for j in range(itr):
        print('Iteration=',j,"Algo.: ", Classifier_name[i])
        X_train, X_test, y_train, y_test = train_test_split(X_new,y)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pp=classification_report(y_test,y_pred, output_dict=True)
        R1+=pp['accuracy']
      Results.append(R1)
      i+=1
    Results_clf_itr.append(Results)
    pickle.dump([Results,PCA_itr], open("E:/FruitLeafDataset.dat", "wb"))
   
    for q in Results_clf_itr:
      print(q)
Matrix_results_itr_Feature=np.array(Results_clf_itr)
print("Done.... all iterations")

##Ensemble of classifiers....

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report

clf1=SVC(kernel="linear", C=0.025)
clf2=MLPClassifier(alpha=1, max_iter=1000)
clf3=LogisticRegression(multi_class='multinomial', random_state=1)

EL_itr=16
eclf1 = VotingClassifier(estimators=[
        ('svm', clf1), ('mlp', clf2), ('lr', clf3)], voting='hard')

R1=0;R2=0;R3=0;R4=0;R5=0;R6=0;R7=0;R8=0;R9=0
previous_pred=[]
for i in range(EL_itr):
    print("Ensemble:[itr=",i,"/",(EL_itr-1))
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    eclf1 = eclf1.fit(X_train, y_train)
    current_pred=eclf1.predict(X_test)

    y_pred=current_pred
    y_true=y_test

    pp=classification_report(y_true,y_pred, output_dict=True)
    print("Accuracy on unknown data is",classification_report(y_test,y_pred))
   
    R1+=pp['accuracy']
    R2+=pp['macro avg']['precision']
    R3+=pp['macro avg']['recall']
    R4+=pp['macro avg']['f1-score']
    R5+=pp['macro avg']['support']
   
    R6+=pp['weighted avg']['precision']
    R7+=pp['weighted avg']['recall']
    R8+=pp['weighted avg']['f1-score']
    R9+=pp['weighted avg']['support']
    current_acc=pp['accuracy']
   
    if i==0:
            current_best_pred=current_pred
            previous_best_acc=current_acc
   
    if current_acc>previous_best_acc:
       current_best_pred=current_pred
y_best_true=y_test
       
R=[R1/EL_itr,R2/EL_itr,R3/EL_itr,R4/EL_itr, R6/EL_itr,R7/EL_itr,R8/EL_itr]
Matrix_results_itr_Feature=np.array(R)

print("Ensemble Learning done here....")

## ploting confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

y_pred=current_best_pred
y_true=y_best_true

CATEGORIES = ['AnanasComosus',
              'AnnonaReticulata',
              'AnnonaSquamosa',
              'ArtocarpusHeterophyllus',
              'ArtocarpusLacucha',
              'CeltisSinensis',
              'CitrusAurantiifolia',
              'CitrusLimon',
              'CitrusMaxima',
              'CitrusPyriformis',
              'CordiaMyxa',
              'FicusRacemosa',
              'GrewiaAsiatica',
              'IpomoeaBatatas',
              'LimoniaAcidissima',
              'LitchiChinensis',
              'MadhucaLongifolia',
              'MangiferaIndica',
              'ManilkaraHexandra',
              'ManilkaraZapota',
              'MorusAlba',
              'NeolamarckiaCadamba',
              'PithecellobiumDulce'
              'PsidiumGuajava',
              'PunicaGranatum',
              'SolanumIycopersicum',
              'SolanumNigrum',
              'SyzygiumJambos',
              'TamarindusIndica',
              'ZizphusMauritiana']
print("ploting confusion matrix....")
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *100, index = [i for i in CATEGORIES],
                     columns = [i for i in CATEGORIES])
plt.figure(figsize = (27,18))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')

from sklearn.metrics import confusion_matrix
print('confusion_matrix ')
print('----------------------')
print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import classification_report
print('classification_report ')
print('----------------------')
print(classification_report(y_test,y_pred))
