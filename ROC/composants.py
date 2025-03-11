#%%
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import roc_auc_score


composants = pd.read_excel("composants_elec.xlsx").dropna()

#%%
n = composants.shape[0]
p = composants.shape[1]
plt.figure()
plt.figure()
for i in range(1, p):
    if composants.columns[i] == "Etat":
        continue
    composants.boxplot(column=composants.columns[i], by='Etat')
    plt.title(f'Boxplot de {composants.columns[i]} par Etat')
    plt.suptitle('')
plt.show()
#composants.boxplot(by="Etat")
moy = composants.groupby('Etat').mean()
print(moy)

#%%
le = LabelEncoder()
Code = le.fit_transform(composants['Etat'])
Code = np.array([e ^ 1 for e in Code])
pd.plotting.scatter_matrix(composants.iloc[:, 0:-1], c=Code)

#%%
C = pd.DataFrame.corr(composants.iloc[:,0:-1], method='pearson')
print(C)
plt.figure()
sns.heatmap(C, vmin=-1, vmax=1, cmap='coolwarm', annot=True)
#%%
#Traçons les individus dans le nouveau repère

plt.figure()
adl = LinearDiscriminantAnalysis()
composantsald = adl.fit_transform(composants.iloc[:,0:-1],Code)
Ra = np.random.randn(n)
print(adl.explained_variance_ratio_)
#plt.scatter(composantsald[:,0],Ra,c=Code) 
plt.scatter(composantsald[:,0],Code,c=Code) 
plt.title("Représentation des données en fonction Code et Code")
plt.show()

#%%
#Traçons la courbe roc sur les variables
plt.figure()
for i in range(0,p-1):
    #Score c'est la variable pour notre truc
    fpr, tpr, thresholds = metrics.roc_curve(Code, composants.iloc[:,i],pos_label=1)
    auc = roc_auc_score(Code,composants.iloc[:,i])
    if auc < 0.5:
        fpr, tpr, thresholds = metrics.roc_curve(Code, -composants.iloc[:,i],pos_label=1)
    plt.plot(fpr,tpr,label=composants.columns[i])
plt.legend(fontsize='small',loc="lower right")
plt.show()
