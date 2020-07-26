#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:04:25 2018

@author: yigit
"""

#%% Gerekli Kütüphanelerin Eklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix # confusion matrisi kütüphanesi

#%% Verinin okunması ve dataframe ile acılması
data = pd.read_csv('voice.csv')

#%% Verinin İncelenmesi
print(data.columns) # Verimizin sütunlarını gösterir
print("---------------------------------------------------------------------")
print(data.shape) # Verimizin boyutunu gosterir (satır,sutun)
print("---------------------------------------------------------------------")
print(data.info()) # Verimizin temel özelliklerini,veri tiplerini,girdileri gösterir
print("---------------------------------------------------------------------")
print(data.describe()) # Verimizin her sutunun istatistiksel özetini çıkarır
print("---------------------------------------------------------------------")
print(data.head()) # Verimizin ilk beş kaydına ait degerleri gösterir
print("---------------------------------------------------------------------")
print(data.tail()) # Verimizin son beş kaydına ait degerleri gösterir
#head ve tail fonksiyonlarına gosterilecek kayıt parametresi verilebilir

#%% Modelin x ve y degiskenlerinin belirlenmesi
# Male = 1 , Female = 0

#Verimizin label sutunu male=1 female = 0 olmak üzere değiştirildi.
data.label = [1 if each =="male" else 0 for each in data.label]

y = data.label.values # boyutu 3168, olarak gözüküyor
y = data.label.values.reshape(-1,1) # boyutu 3168,1 olarak düzelttik

# x_data degiskenine verinin son sutunu silinip geri kalan kısmı atandı
x_data = data.drop(["label"],axis=1)
# Degisiklik veri üzerinde yapılmadı,eger yapılmak istenseydi inplace=true yazılmalı
#axis = 0 satır silinmesi , axis=1 sutun silinmesi için kullanılır.

#%% Normalization
# Degiskenlerin degerlerinin birbiri üstünde etki kurup etkilememesi için bu
#islem yapılır
x = (x_data-np.min(x_data)) /(np.max(x_data)-np.min(x_data)).values
#Min-Max normalization yontemi kullanılarak 0-1 arasında ölçeklendirme yapıldı

#%% Verinin Train ve Test olarak ayrılması
from sklearn.model_selection import train_test_split
#scikit kütüphanesi makine öğrenmesi algoritmaları için kullanılır
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=5)
#Train ve test verileri oluşturuldu test boyutu %20 olarak belirlendi
#random_state random id olarak kullanılır

#%% Logistic Regression ile Sınıflandırma
from sklearn.linear_model import LogisticRegression #Kütüphanenin eklenmesi

lr = LogisticRegression() # Algoritma modelinin oluşturulması
lr.fit(x_train,y_train) # Modelin eğitilmesi
lr_predictions = lr.predict(x_test) # Algoritmanın tahmin sonuçları
LR_accuracy = lr.score(x_test,y_test) # Algoritmanın doğruluğu
cm_lr = confusion_matrix(y_test,lr_predictions) # confusion matrisi
LR_sensitivity = cm_lr[0,0]/(cm_lr[0,0]+cm_lr[0,1]) #sensitivity hesaplanması
LR_specificity = cm_lr[1,1]/(cm_lr[1,0]+cm_lr[1,1]) #specificity hesaplanması

#%% KNN Algoritması ile Sınıflandırma
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5) # n_neighbors k değeri demek
knn.fit(x_train,y_train) # Modelin eğitilmesi
knn_predictions = knn.predict(x_test) # Algoritmanın tahmin sonuçları
KNN_accuracy = knn.score(x_test,y_test) # Algoritmanın doğruluğu
cm_knn = confusion_matrix(y_test,knn_predictions) # confusion matrisi
KNN_sensitivity = cm_knn[0,0]/(cm_knn[0,0]+cm_knn[0,1]) #sensitivity hesaplanması
KNN_specificity = cm_knn[1,1]/(cm_knn[1,0]+cm_knn[1,1])

# En iyi k değerinin seçilmesi
skorlist = []
for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors = i) # n_neighbors k değeri demek
    knn.fit(x_train,y_train)
    skorlist.append(knn.score(x_test,y_test))
    
plt.plot(range(1,15),skorlist,color="red",label="K değer seçimi")
plt.legend()
plt.xlabel("k değerleri")
plt.ylabel("Doğruluk oranı")
plt.title("En Uygun K Değeri Seçimi")
plt.show()

print("En iyi k değeri :",skorlist.index(max(skorlist))+1)

#%% SVM Algoritması ile Sınıflandırma
from sklearn.svm import SVC # Gerekli kütüphanenin eklenmesi

svm = SVC(random_state = 1) # Modelin oluşturulması ve random_id = 1
svm.fit(x_train,y_train) # Modelin eğitilmesi
svm_predictions = svm.predict(x_test)
SVM_accuracy = svm.score(x_test,y_test) # Algoritmanın doğruluğu
cm_svm = confusion_matrix(y_test,svm_predictions) # confusion matrisi
SVM_sensitivity = cm_svm[0,0]/(cm_svm[0,0]+cm_svm[0,1]) #sensitivity hesaplanması
SVM_specificity = cm_svm[1,1]/(cm_svm[1,0]+cm_svm[1,1]) #specificity hesaplanması

#%% Naive Bayes Algoritması ile Sınıflandırma
from sklearn.naive_bayes import GaussianNB # Gerekli kütüphanenin eklenmesi

nb = GaussianNB() # Modelin Oluşturulması
nb.fit(x_train,y_train) # Modelin eğitilmesi
nb_predictions = nb.predict(x_test)
NB_accuracy = nb.score(x_test,y_test) # Algoritmanın doğruluğu
cm_nb = confusion_matrix(y_test,nb_predictions) # confusion matrisi
NB_sensitivity = cm_nb[0,0]/(cm_nb[0,0]+cm_nb[0,1]) #sensitivity hesaplanması
NB_specificity = cm_nb[1,1]/(cm_nb[1,0]+cm_nb[1,1]) #specificity hesaplanması

#%% Decision Tree Algoritması ile Sınıflandırma 
from sklearn.tree import DecisionTreeClassifier #Gerekli kütüphanenin eklenmesi

dt = DecisionTreeClassifier() # Modelin Oluşturulması
dt.fit(x_train,y_train) # Modelin eğitilmesi
dt_predictions = dt.predict(x_test)
DT_accuracy = dt.score(x_test,y_test) # Algoritmanın doğruluğu
cm_dt = confusion_matrix(y_test,dt_predictions) # confusion matrisi
DT_sensitivity = cm_dt[0,0]/(cm_dt[0,0]+cm_dt[0,1]) #sensitivity hesaplanması
DT_specificity = cm_dt[1,1]/(cm_dt[1,0]+cm_dt[1,1]) #specificity hesaplanması

#%% Random Forest Algoritması ile Sınıflandırma 
from sklearn.ensemble import RandomForestClassifier # Kütüphanenin eklenmesi

rf = RandomForestClassifier(n_estimators=50,random_state=5)
#n_estimators = algoritmadaki ağaç yapısı sayısı random_state = random id
rf.fit(x_train,y_train) # Modelin eğitilmesi
rf_predictions = rf.predict(x_test)
RF_accuracy = rf.score(x_test,y_test) # Algoritmanın doğruluğu
cm_rf = confusion_matrix(y_test,rf_predictions) # confusion matrisi
RF_sensitivity = cm_rf[0,0]/(cm_rf[0,0]+cm_rf[0,1]) #sensitivity hesaplanması
RF_specificity = cm_rf[1,1]/(cm_rf[1,0]+cm_rf[1,1]) #specificity hesaplanması

#%% Confusion Matrisi Oluşturma ve Görselleştirme

import seaborn as sns
f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm_knn,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_prediction")
plt.ylabel("y_true")
plt.title("KNN Confusion Matrix")
plt.show()

#%% Model Selection ile Modelin Doğruluğunu Kontrol Etme
# K-Fold Cross Validation Yontemi Kullanılacak 
from sklearn.model_selection import cross_val_score # Kütüphane
# K-Fold accuracy listesi 10 eğitim için
accuracies = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=10).reshape(-1,1)
k_fold_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print("Average accuracy: ",k_fold_accuracy) # Ortalama model doğruluğu
print("Acerage std : ",std_accuracy )# Modelin standart sapması

#%% Algoritmaların ve Modelin Doğruluklarının Karşılaştırılması
accuracy_list = ["LR","KNN","SVM","NB","DT","RF","K-Fold"]
accuracy_results_list =[LR_accuracy,KNN_accuracy,SVM_accuracy,NB_accuracy,DT_accuracy,RF_accuracy,k_fold_accuracy]

for i in range(len(accuracy_results_list)):
    accuracy_results_list[i] = accuracy_results_list[i]*100
    
plt.figure(3,figsize=(13,4))
plt.subplot(231)
plt.bar(accuracy_list,accuracy_results_list,color="blue")
plt.legend()
plt.title("Accuracy List")
plt.xlabel("Algoritmalar")
plt.ylabel("Accuracy(%)")
plt.show()

#%% Algoritmaların Sensitivity Oranlarının Karşılaştırılması
sensitivity_list = ["LR","KNN","SVM","NB","DT","RF"]
sensitivity_results_list = [LR_sensitivity,KNN_sensitivity,SVM_sensitivity,NB_sensitivity,DT_sensitivity,RF_sensitivity]

for i in range(len(sensitivity_results_list)):
    sensitivity_results_list[i] = sensitivity_results_list[i]*100

plt.figure(3,figsize=(13,4))
plt.subplot(232)
plt.bar(sensitivity_list,sensitivity_results_list,color="red")
plt.legend()
plt.title("Sensitivity List")
plt.xlabel("Algoritmalar")
plt.ylabel("Sensitivity(%)")
plt.show()

#%% Algoritmaların Specificity Oranlarının Karşılaştırılması
specificity_list = ["LR","KNN","SVM","NB","DT","RF"]
specificity_results_list = [LR_specificity,KNN_specificity,SVM_specificity,NB_specificity,DT_specificity,RF_specificity]

for i in range(len(specificity_results_list)):
    specificity_results_list[i] = specificity_results_list[i]*100

plt.figure(3,figsize=(13,4))
plt.subplot(233)
plt.bar(specificity_list,specificity_results_list,color="green")
plt.legend()
plt.title("Specificity List")
plt.xlabel("Algoritmalar")
plt.ylabel("specificity(%)")
plt.show()

#%% Roc Curve
# KNN Algoritması için roc eğrisinin çizimi
from sklearn.metrics import roc_curve, auc
probs = knn.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

#Roc egrisinin gorsellestirilmesi
plt.figure(4)
plt.title('KNN ROC Grafiği')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%% Principle Component Analysis
from sklearn.decomposition import PCA

pca = PCA(n_components = 2, whiten = True) #whiten=normalize islemi n=boyut sayısı
pca.fit(x) # modeli eğit
x_pca = pca.transform(x) # eğitilen modeli yeni değişkene aktar

print("Variance ratio : ",pca.explained_variance_ratio_) # bileşenlerin oranı
print("Total :", sum(pca.explained_variance_ratio_)) # toplam bileşenlerin değeri

# 2d Gorselleştirme

data["p1"] = x_pca[:,0] # Dataya yeni sütun eklendi
data["p2"] = x_pca[:,1] # Dataya yeni sütun eklendi

color = ["red","green"]
labels=["Female","Male"]
import matplotlib.pyplot as plt

plt.figure(5)
for each in range(2):
    plt.scatter(data.p1[data.label==each],data.p2[data.label==each],color=color[each],label=labels[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.title("Principle Component Analysis")
plt.show()

#%% Görselleştirme
import matplotlib.pyplot as plt
        
labels = list(data.label.unique())

for i in range(0,len(labels)):
    if labels[i] == 1:
        labels[i] = 'Erkek'
    else:
        labels[i] = 'Kadın'
        
female_ratio = len(data[data['label'] == 0]) / len(data) * 100
male_ratio = len(data[data['label'] == 1]) / len(data) * 100


sizes = [female_ratio,male_ratio]
colors = ['red','blue']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=180)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.title('Seslerin Cinsiyete Dağılımı')
plt.show()
