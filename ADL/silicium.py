#%%
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

silicium_raw = pd.read_excel("defauts_silicium_v2.xlsx",index_col=0)
silicium_remain = silicium_raw.dropna()[149:]
silicium = silicium_raw.dropna()[:149]
print(silicium)
couleur = {"Inclusion":"blue","Rayure":"green","Contamination":"purple"}

#Les données des fleurs sont triées par groupe
#Si l'on fait un boxplot ou triera par groupe ce qui affichera les moyennes pour chaque groupe par lequel on groupby
#%%
n = silicium.shape[0]
p = silicium.shape[1]
plt.figure()
plt.figure()
for i in range(1, p):
   silicium.boxplot(column=silicium.columns[i], by='Type')
   plt.title(f'Boxplot de {silicium.columns[i]} par Type')
   plt.suptitle('')
plt.show()
#silicium.boxplot(by="Type")
moy = silicium.groupby('Type').mean()
print(moy)

#%%
le = LabelEncoder()
silicium['Code'] = le.fit_transform(silicium['Type'])
pd.plotting.scatter_matrix(silicium.iloc[:, 1:-1], c=silicium["Code"])

#%%
#Traçons les individus dans le nouveau repère

plt.figure()
adl = LinearDiscriminantAnalysis()
siliciumald = adl.fit_transform(silicium.iloc[:,1:-1],silicium['Code'])
plt.scatter(siliciumald[:,0],siliciumald[:,1],c=silicium["Code"])    

inconnu = adl.transform(silicium_remain.iloc[:,1:])
plt.scatter(inconnu[:,0], inconnu[:,0], color = "black")
#plt.annotate(silicium["Group"][i],(irisald[i,0],irisald[i,1]))
plt.title("Représentation des données en fonction de CP1 et CP2")
plt.ylabel("CPD2")
plt.xlabel("CPD1")
plt.show()


#%%



print("=============\n\n")
#print(np.sum(silicium.var())) # A régler

#Cercle
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
circle = plt.Circle((0, 0), 1, color='b', fill=False)
ax.add_artist(circle)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.grid()
ax.set_aspect('equal', adjustable='box')


for i in range(1,p):
    CP1 = np.corrcoef(silicium.iloc[:,i],siliciumald[:,0])[0,1]
    CP2 = np.corrcoef(silicium.iloc[:,i],siliciumald[:,1])[0,1]
    if silicium.columns[i] != "Code" and silicium.columns[i] != "Type":
        plt.scatter(CP1,CP2)
        plt.annotate(silicium.columns[i],(CP1,CP2))
plt.title('Cercle Unitaire')
plt.show()

    