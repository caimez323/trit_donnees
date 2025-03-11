#%%
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


iris = datasets.load_iris()
irisData= pd.DataFrame(iris.data,columns=iris.feature_names) 
irisData['Group']=iris.target
#print(irisData)

#%%
n = irisData.shape[0]
p = irisData.shape[1]
moy = irisData.groupby('Group').mean()
#print(moy)


#%%
#Stratify permet de respecter la proportion d'individu au sein de chaque groupe
X_train, X_test, y_train, y_test = train_test_split(iris.data , iris.target, test_size=1/3 , stratify = iris.target)
adl = LinearDiscriminantAnalysis()
irisTrain = adl.fit(X_train,y_train)

#On est critique sur le nombre d'individus
#print("Répartition d'apprentissage : {}.\n".format(y_train.value_counts()))
#print("Répartition des tests : {}.\n".format(y_test.value_counts()))

#Prediction sur l'échantillon de test
y_pred = adl.predict(X_test)

#La réponse du train est dans y_test
#Le résultat du pred est dans y_pred
precision = adl.score(X_test, y_test)

#Bien mais pas très détaillé
print("Précision de {}%".format(100*sum(y_pred == y_test)/len(y_test)))

conf = confusion_matrix(y_test, y_pred,normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=conf)
disp.plot()
plt.show()

