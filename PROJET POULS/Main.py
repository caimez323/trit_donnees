#%%
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import scipy.io 
import seaborn as sns
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import roc_auc_score


# %%
data = scipy.io.loadmat('./data.mat') 
df1 = pd.DataFrame(data['data1'])
df1.columns = [
    "Numéro du pulse",
    "Classement cardiologue",
    "Début du pulse",
    "Fin du pulse",
    "Max intercorrélation",
    "Différence max",
    "Différence min",
    "Diff pression brassard",
    "Diff largeur"
]
df1["Numéro du pulse"] = df1["Numéro du pulse"].astype(int).astype(str)

df2 = pd.DataFrame(data['data2'])
df2.columns = [
    "Numéro du pulse",
    "Classement cardiologue",
    "Début du pulse",
    "Fin du pulse",
    "Max intercorrélation",
    "Différence max",
    "Différence min",
    "Diff pression brassard",
    "Diff largeur"
]
df2["Numéro du pulse"] = df2["Numéro du pulse"].astype(int).astype(str)

df3 = pd.DataFrame(data['data3'])
df3.columns = [
    "Numéro du pulse",
    "Classement cardiologue",
    "Début du pulse",
    "Fin du pulse",
    "Max intercorrélation",
    "Différence max",
    "Différence min",
    "Diff pression brassard",
    "Diff largeur"
]
df3["Numéro du pulse"] = df3["Numéro du pulse"].astype(int).astype(str)

df1 = df1.drop(columns=["Début du pulse", "Fin du pulse"])
df2 = df2.drop(columns=["Début du pulse", "Fin du pulse"])
df3 = df3.drop(columns=["Début du pulse", "Fin du pulse"])

# %%
# Suppose que ton DataFrame est déjà chargé et nommé `df`

# Convertir la colonne "Classement cardiologue" en label encodé
le = LabelEncoder()
df1['Code'] = le.fit_transform(df1['Classement cardiologue'])

# Nombre de colonnes/features
n = df1.shape[0]
p = df1.shape[1]

# Tracer les boxplots par rapport à la variable qualitative "Classement cardiologue"
plt.figure(figsize=(12, 6))
for i in range(1, p):
    col = df1.columns[i]
    if "Classement cardiologue" in col or "Numéro du pulse" in col or "Code" in col:  # Skip colonne qualitative & ID
        continue
    plt.figure()
    df1.boxplot(column=col, by='Classement cardiologue')
    plt.title(f'Boxplot de {col} par Classement cardiologue')
    plt.suptitle('')  # Pour enlever le titre automatique
    plt.xlabel('Classement cardiologue')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# Moyenne par classe
# Exclude non-numeric columns before calculating the mean
numeric_columns = df1.select_dtypes(include=[np.number]).columns
moy = df1.groupby('Classement cardiologue')[numeric_columns].mean()
print(moy)

#%%
C = pd.DataFrame.corr(df1.iloc[:,2:-1], method='pearson')
print(C)
plt.figure()
sns.heatmap(C, vmin=-1, vmax=1, cmap='coolwarm', annot=True)

#%%
from sklearn.decomposition import PCA

# On enlève les 2 premières colonnes et la dernière
# Supposons qu'on ait déjà nettoyé df1 en enlevant les colonnes "Début du pulse" et "Fin du pulse"
df1_reduit = df1.iloc[:, 2:-1]

# 1. Matrice de covariance
Cov = df1_reduit.cov()
print("Matrice de covariance:")
print(Cov)

print("=============\n\n")

# 2. Application de l'ACP
acp = PCA()
Xacp = acp.fit_transform(df1_reduit)  # On effectue l'ACP

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
print("Variance totale : ", np.sum(df1_reduit.var()))
print("Somme des variances expliquées par l'ACP : ", np.sum(acp.explained_variance_))

# 5. Scatter plot des individus dans le nouveau repère (2 premières composantes)
plt.figure(figsize=(8, 6))
for i in range(len(Xacp)):
    plt.annotate(df1.index[i], (Xacp[i, 0], Xacp[i, 1]))
plt.scatter(Xacp[:, 0], Xacp[:, 1], c='blue', marker='o')
plt.title('Projection des individus dans le nouveau repère')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.grid(True)
plt.show()

# %%
#Traçons les individus dans le nouveau repère
#On fait une ADL
plt.figure()
adl = LinearDiscriminantAnalysis()
composantsald = adl.fit_transform(df1.iloc[:,1:-1],df1["Code"])
Ra = np.random.randn(n)
print(adl.explained_variance_ratio_)
plt.scatter(composantsald[:,0],Ra,c=df1["Code"]) 
#plt.scatter(composantsald[:,0],df1["Code"],c=df1["Code"]) 
plt.title("Représentation des données en fonction Code et Code")
plt.show()
# %%
# On fait l'analyse ROC à partir de l'ADL


plt.figure()
for i in range(2,df1.shape[1]-1):
    #Score c'est la variable pour notre truc
    fpr, tpr, thresholds = metrics.roc_curve(df1["Code"], df1.iloc[:,i],pos_label=1)
    auc = roc_auc_score(df1["Code"],df1.iloc[:,i])
    if auc < 0.5:
        fpr, tpr, thresholds = metrics.roc_curve(df1["Code"], -df1.iloc[:,i],pos_label=1)
    plt.plot(fpr,tpr,label=df1.columns[i])
plt.legend(fontsize='small',loc="lower right")
plt.show()

# %%
