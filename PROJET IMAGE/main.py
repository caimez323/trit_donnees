#%%
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import scipy.io 
import seaborn as sns
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np
import scipy.stats as stats

img_data = pd.read_excel("database.xlsx",index_col=0)
img_data = img_data.dropna()

print(img_data)

#%%
n = img_data.shape[0]
p = img_data.shape[1]
"""
plt.figure()
for i in range(2, p):
   img_data.boxplot(column=img_data.columns[i], by='SousClasse')
   plt.title(f'Boxplot de {img_data.columns[i]} par SousClasse')
   plt.suptitle('')
plt.show()
#silicium.boxplot(by="Type")
moy = img_data.groupby('SousClasse').mean()
print(moy)
"""
#%%

le = LabelEncoder()
img_data['Code'] = le.fit_transform(img_data['Classe'])


#%%
from sklearn.decomposition import PCA

# On enlève les 2 premières colonnes et la dernière
# Supposons qu'on ait déjà nettoyé img_data en enlevant les colonnes "Début du pulse" et "Fin du pulse"
img_data_reduit = img_data.iloc[:, 2:-1]

# 1. Matrice de covariance
Cov = img_data_reduit.cov()
print("Matrice de covariance:")
print(Cov)

print("=============\n\n")

# 2. Application de l'ACP
acp = PCA()
Xacp = acp.fit_transform(img_data_reduit)  # On effectue l'ACP

# Affichage des résultats ACP
print("Composantes principales (Xacp) :\n", Xacp)
print("Variance de chaque axe : {}".format(acp.explained_variance_))
print("Ratio de variance de chaque axe : {}".format(acp.explained_variance_ratio_))

# Affichage des composants
comp = acp.components_
print("Composantes : \n{}\n".format(comp))

# 3. Graphique de la variance expliquée
plt.figure(figsize=(10, 6))
plt.bar(np.arange(1, len(acp.explained_variance_ratio_) + 1), acp.explained_variance_ratio_)
plt.plot(np.arange(1, len(acp.explained_variance_ratio_) + 1), np.cumsum(acp.explained_variance_ratio_))
plt.ylabel("Variance expliquée en ratio et cumul")
plt.xlabel("Nombre de facteurs")
plt.title("Variance expliquée par chaque composante principale")
plt.show()

# 4. Affichage de la projection des individus dans le nouveau repère
print("Variance totale : ", np.sum(img_data_reduit.var()))
print("Somme des variances expliquées par l'ACP : ", np.sum(acp.explained_variance_))

# 5. Scatter plot des individus dans le nouveau repère (2 premières composantes)
plt.figure(figsize=(8, 6))
for i in range(len(Xacp)):
    plt.annotate(img_data.index[i], (Xacp[i, 0], Xacp[i, 1]))
plt.scatter(Xacp[:, 0], Xacp[:, 1], c='blue', marker='o')
plt.title('Projection des individus dans le nouveau repère')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.grid(True)
plt.show()

# %%
#Traçons les individus dans le nouveau repère
#On fait une ADL
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
plt.figure()
adl = LinearDiscriminantAnalysis()
composantsald = adl.fit_transform(img_data.iloc[:,2:-1],img_data["Code"])
Ra = np.random.randn(n)
print(adl.explained_variance_ratio_)
plt.scatter(composantsald[:,0],Ra,c=img_data["Code"]) 
#plt.scatter(composantsald[:,0],img_data["Code"],c=img_data["Code"]) 
plt.title("Représentation des données en fonction Code et Code")
#Ajouter legende
plt.show()

#%%



