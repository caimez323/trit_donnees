#%%
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
# On peut ensuite se ramener à une DataFrame
irisData= pd.DataFrame(iris.data,columns=iris.feature_names) 
irisData['Group']=iris.target
print(irisData)

#Les données des fleurs sont triées par groupe
#Si l'on fait un boxplot ou triera par groupe ce qui affichera les moyennes pour chaque groupe par lequel on groupby
#%%
n = irisData.shape[0]
p = irisData.shape[1]
irisData.boxplot(by="Group")
moy = irisData.groupby('Group').mean()
print(moy)

#%%
pd.plotting.scatter_matrix(irisData.iloc[:, 0:4], c=irisData['Group'])

#%%
#Traçons les individus dans le nouveau repère

adl = LinearDiscriminantAnalysis()
irisald = adl.fit_transform(irisData.iloc[:,0:4],irisData['Group'])
couleur = {0:"blue",1:"green",2:"purple"}
plt.figure()
for i in range(len(irisData)):
    plt.scatter(irisald[i,0],irisald[i,1],color=couleur[irisData["Group"][i]])    
    #plt.annotate(irisData["Group"][i],(irisald[i,0],irisald[i,1]))
plt.title("Représentation des données en fonction de CP1 et CP2")
plt.ylabel("CP2")
plt.xlabel("CP1")
plt.show()


#%%



print("=============\n\n")
print(np.sum(irisData.var()))

#Cercle
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
circle = plt.Circle((0, 0), 1, color='b', fill=False)
ax.add_artist(circle)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.grid()
ax.set_aspect('equal', adjustable='box')


for i in range(p):
    CP1 = np.corrcoef(irisData.iloc[:,i],irisald[:,0])[0,1]
    CP2 = np.corrcoef(irisData.iloc[:,i],irisald[:,1])[0,1]
    if irisData.columns[i] != "Group":
        plt.scatter(CP1,CP2)
        plt.annotate(irisData.columns[i],(CP1,CP2))
plt.title('Cercle Unitaire')
plt.show()

    