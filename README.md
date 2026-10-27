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

Deep Learning Algorithms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])
data_transforms = {
'train': transforms.Compose([
transforms.RandomResizedCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(mean, std)
]),
'val': transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean, std)
]),}
test_transforms=transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean, std)
])
#Test_path='C:/Users/HP/Desktop/DD/test'
#Test_path='C:/Users/HP/Desktop/SeedsV-22_train_valid_test/SeedsV-22_Test'
Test_path=r'D:\Dataset\DatasetAgumented\test'
#Test_path='C:/Users/HP/Desktop/XX/test'
#data_dir = 'C:/Users/HP/Desktop/SeedsV-22_train_valid_test/Data/'
#data_dir='C:/Users/HP/Desktop/DD/data'
#data_dir='C:/Users/HP/Desktop/SeedsV-28'
data_dir=r'D:\Dataset\DatasetAgumented\data'
#data_dir='C:/Users/HP/Desktop/XX/Data'
batch=24
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
data_transforms[x])
for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=shuffle=True, num_workers=
for x in ['train', 'val']}
test_dataloaders=DataLoader(torchvision.datasets.ImageFolder(Test_path,transform=batch_size=batch,shuffle=True
)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(class_names)
num_classes=len(class_names)
'''
def imshow(inp, title):
"""Imshow for Tensor."""
inp = inp.numpy().transpose((1, 2, 0))
inp = std * inp + mean
inp = np.clip(inp, 0, 1)
plt.imshow(inp)
plt.title(title)
plt.show()
'''
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
#imshow(out, title=[class_names[x] for x in classes])
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
epoch_acc_train=[]
epoch_loss_train=[]
epoch_acc_valid=[]
epoch_loss_valid=[]
for epoch in range(num_epochs):
print('Epoch {}/{}'.format(epoch, num_epochs - 1))
print('-' * 10)
# Each epoch has a training and validation phase
for phase in ['train', 'val']:
if phase == 'train':
model.train() # Set model to training mode
else:
model.eval() # Set model to evaluate mode
running_loss = 0.0
running_corrects = 0
# Iterate over data.
for inputs, labels in dataloaders[phase]:
inputs = inputs.to(device)
labels = labels.to(device)
# forward
# track history if only in train
with torch.set_grad_enabled(phase == 'train'):
outputs = model(inputs)
_, preds = torch.max(outputs, 1)
loss = criterion(outputs, labels)
# backward + optimize only if in training phase
if phase == 'train':
optimizer.zero_grad()
loss.backward()
optimizer.step()
# statistics
running_loss += loss.item() * inputs.size(0)
running_corrects += torch.sum(preds == labels.data).data.if phase == 'train':
scheduler.step()
epoch_loss = running_loss / dataset_sizes[phase]
epoch_acc = running_corrects/ dataset_sizes[phase]
print('{} Loss: {:.4f} Acc: {:.4f}'.format(
phase, epoch_loss, epoch_acc))
if phase=='train':
epoch_acc_train.append(epoch_acc)
epoch_loss_train.append(epoch_loss)
else:
epoch_acc_valid.append(epoch_acc)
epoch_loss_valid.append(epoch_loss)
# deep copy the model
if phase == 'val' and epoch_acc > best_acc:
best_acc = epoch_acc
best_model_wts = copy.deepcopy(model.state_dict())
print()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
#train_results=[epoch_acc_train,epoch_loss_train]
#valid_results=[epoch_acc_valid,epoch_loss_valid]
# load best model weights
model.load_state_dict(best_model_wts)
return ([model,epoch_acc_train,epoch_loss_train, epoch_acc_valid,epoch_#%%
def test_model(model, test_dataloaders,criterion, optimizer):
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else
opCount=0
model.eval() # Set model to training mode
running_loss = 0.0
running_corrects = 0
y_true=[]
y_pred=[]
for inputs, labels in test_dataloaders:
lbl=labels
inputs = inputs.to(device)
labels = labels.to(device)
optimizer.zero_grad()
with torch.set_grad_enabled(mode=False):
outputs = model(inputs)
_, preds = torch.max(outputs, 1)
loss = criterion(outputs, labels)
opCount+=len(outputs)
running_loss += loss.item() * inputs.size(0)
running_corrects += torch.sum(preds == labels.data)
outputs_x = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().y_pred.extend(outputs_x)
y_true.extend(lbl)
test_loss = running_loss / opCount
test_acc = running_corrects / opCount
test_acc=test_acc.data.cpu().numpy()
y_true=[y_true[i].numpy().tolist() for i in range(len(y_true))]
print('{} Loss: {:.4f} Acc: {:.4f}'.format(
'Test:', test_loss, test_acc))
return ([test_loss,test_acc,y_true, y_pred])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Deep learning models setting
def set_parameter_requires_grad(model, feature_extracting):
if feature_extracting:
for param in model.parameters():
param.requires_grad = False
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=
# Initialize these variables which will be set in this if statement. # variables is model specific.
model_ft = None
input_size = 0
if model_name == "resnet":
""" Resnet18
"""
model_ft = models.resnet18(pretrained=use_pretrained)
set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
input_size = 224
elif model_name == "alexnet":
""" Alexnet
"""
model_ft = models.alexnet(pretrained=use_pretrained)
set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
input_size = 224
elif model_name == "vgg":
""" VGG11_bn
"""
model_ft = models.vgg11_bn(pretrained=use_pretrained)
set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
input_size = 224
elif model_name == "squeezenet":
""" Squeezenet
"""
model_ft = models.squeezenet1_0(pretrained=use_pretrained)
set_parameter_requires_grad(model_ft, feature_extract)
model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(
model_ft.num_classes = num_classes
input_size = 224
elif model_name == "densenet":
""" Densenet
"""
model_ft = models.densenet121(pretrained=use_pretrained)
set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, num_classes)
input_size = 224
elif model_name == "inception":
""" Inception v3
Be careful, expects (299,299) sized images and has auxiliary output
"""
model_ft = models.inception_v3(pretrained=use_pretrained)
set_parameter_requires_grad(model_ft, feature_extract)
# Handle the auxilary net
num_ftrs = model_ft.AuxLogits.fc.in_features
model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
# Handle the primary net
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,num_classes)
input_size = 299
elif model_name=="googlenet":
model_ft=models.googlenet(pretrained=True)
input_size = 224
elif model_name=="shufflenet":
model_ft=models.shufflenet_v2_x1_0(pretrained=use_pretrained)
input_size = 224
elif model_name=="resnext50":
model_ft=models.resnext50_32x4d(pretrained=use_pretrained)
input_size = 224
else:
print("Invalid model name, exiting...")
exit()
return model_ft, input_size
#%% Initialize the model for this run
#, , "inception"
# "resnet", "alexnet","vgg", "squeezenet",
model_name=["resnet", "alexnet","vgg", "squeezenet","googlenet","shufflenet"
Model_complete=[]
train_acc_list=[]
train_loss_list=[]
valid_acc_list=[]
valid_loss_list=[]
test_loss_list=[]
test_acc_list=[]
y_true_list=[]
y_pred_list=[]
for name in model_name:
print("Executing model:",name)
model, input_size = initialize_model(name, num_classes, feature_extract=
#%%%%%%%%%%%%%%%%%%%%%%% setting of the optimizer................
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model,epoch_acc_train,epoch_loss_train, epoch_acc_valid,epoch_loss_valid= test_loss,test_acc,y_true, y_pred=test_model(model, test_dataloaders, Model_complete.append(name)
train_acc_list.append(epoch_acc_train)
train_loss_list.append(epoch_loss_train)
valid_acc_list.append(epoch_acc_valid)
valid_loss_list.append(epoch_loss_valid)
test_loss_list.append(test_loss)
test_acc_list.append(test_acc)
y_true_list.append(y_true)
y_pred_list.append(y_pred)
print("Done....all models in the list")
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#y_pred=current_best_pred
#y_true=y_best_true
CATEGORIES = ['AnanasComosus',
'AnnonaReticulata',
'AnnonaSquamosa',
'ArtocarpusHeterophyllus',
'ArtocarpusLacucha',
'CarissaCarandas',
'CitrusAurantiifolia',
'CitrusLimon',
'CitrusMaxima',
'CitrusPyriformis',
'CordiaMyxa',
'FicusRacemosa',
'GrewiaAsiatica',
'LimoniaAcidissima',
'LitchiChinensis',
'MadhucaLongifolia',
'MangiferaIndica',
'ManilkaraHexandra',
'ManilkaraZapota',
'MorusAlba',
'NeolamarckiaCadamba',
'PithecellobiumDulce',
'PsidiumGuajava',
'PunicaGranatum',
'SolanumIycopersicum',
'SolanumNigrum',
'SyzygiumJambos',
'TamarindusIndica',
'VitisVinifera',
'ZizphusMauritiana']
print("ploting confusion matrix....")
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *100, index = [i for i
columns = [i for i in CATEGORIES])
plt.figure(figsize = (27,18))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')
#ploting confusion matrix....
from sklearn.metrics import confusion_matrix
print('confusion_matrix ')
print('----------------------')
print(confusion_matrix(y_true,y_pred))
#confusion_matrix
#----------------------
#Classification Report
#from sklearn.metrics import classification_report
print('classification_report ')
print('----------------------')
print(classification_report(y_true,y_pred))
