#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Une variable quantitative continue => histogramme 
#       Utilisation de boxplot
#%%
#Une variable discrete => diagramme baton
#       Utilisation de crosstab 
crimes = pd.read_excel("Criminalite.xlsx",index_col=0)
crimes = crimes.dropna()
#CENTRE REDUIT
crimes = (crimes-crimes.mean())/crimes.std()

#crimes.describe()
#crimes.info()
n = crimes.shape[0]
p = crimes.shape[1]
print("Nombres d'individus : {}".format(n))
for d in crimes:
    print("{} : {}".format(d, crimes[d].dtype))
    
#===
print("Analyse Simples (sur un élément)\n=============\n\n")
print("Moyenne : \n")
print(crimes.mean())
print("Variance : \n")
print(crimes.var())

plt.figure()
crimes.boxplot()
plt.show()


#%%

print("Analyse Bivarié (sur deux éléments)\n=============\n\n")
pd.plotting.scatter_matrix(crimes)

#%%
print("Heat map \n=============\n\n")
C = pd.DataFrame.corr(crimes, method='pearson')
print(C)
plt.figure()
sns.heatmap(C, vmin=-1, vmax=1, cmap='coolwarm', annot=True)
#%%

print("=============\n\n")
#Matrice de covariance
Cov = pd.DataFrame.cov(crimes)
print(Cov)

print("=============\n\n")

from sklearn.decomposition import PCA
acp = PCA()
Xacp = acp.fit_transform(crimes) # ou Xacp=acp.fit(X).transform(X)
print(Xacp)
print("Variance de chaque axe : {}".format(acp.explained_variance_))
print("Ratio de variance de chaque axe : {}".format(acp.explained_variance_ratio_))
#Ici on voit que les deux premières valeurs représentent plus de 99% de la variance totales
comp = acp.components_
print("Components : \n{}\n".format(acp.components_))
#Ce grapique permet de voir la somme des variances normalisés
#Il permet de déterminer les facteurs qui ont le plus d'impact et voir jusqu'au quels n on prend

plt.figure()
# Graphique des variances expliquées
plt.bar(np.arange(1, p+1), acp.explained_variance_ratio_)
plt.plot(np.arange(1, p+1), np.cumsum(acp.explained_variance_ratio_))
plt.ylabel("Variance expliquée en ratio et cumul")
plt.xlabel("Nombre de facteurs")
plt.show()
#%%
#Ici on sait qu'on a besoin de 2 facteurs pour expliquer 99% de la variance
#On fait juste un scatter
#La variance totale du nuage de points-individus peut-être calculé comme
#La somme des variances de chaque facteurs
print(np.sum(crimes.var()))
#La somme des vp
print(np.sum(acp.explained_variance_))

print("=============\n\n")
#Traçons les individus dans le nouveau repère

plt.figure()
for i in range(len(Xacp)):
    plt.annotate(crimes.index[i],(Xacp[i,0],Xacp[i,1]))
plt.scatter(Xacp[:, 0], Xacp[:, 1], c='blue', marker='o')
plt.title('Projection des données dans le nouveau repère')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.grid(True)
plt.show()

#Cercle
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
circle = plt.Circle((0, 0), 1, color='b', fill=False)
ax.add_artist(circle)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.grid()
ax.set_aspect('equal', adjustable='box')


for i in range(p):
    CP1 = np.corrcoef(crimes.iloc[:,i],Xacp[:,0])[0,1]
    print(np.corrcoef(crimes.iloc[:,i],Xacp[:,0]))
    CP2 = np.corrcoef(crimes.iloc[:,i],Xacp[:,1])[0,1]
    plt.scatter(CP1,CP2)
    plt.annotate(crimes.columns[i],(CP1,CP2))
plt.title('Cercle Unitaire')
plt.show()
# %%
