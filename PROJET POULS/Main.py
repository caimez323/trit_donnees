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
import numpy as np
import scipy.stats as stats

def intervalle_confiance_95(data):
    data = np.array(data)
    n = len(data)
    moyenne = np.mean(data)
    ecart_type = np.std(data, ddof=1)  # écart-type corrigé
    erreur_type = ecart_type / np.sqrt(n)
    
    # t de Student à 95% de confiance, bilatéral
    t_score = stats.t.ppf(1 - 0.025, df=n - 1)
    
    marge = t_score * erreur_type
    return moyenne - marge, moyenne + marge


#Pour l'oral, il faudra expliquer les outils que l'on utiliser, pourquoi on les utiliser, c'est quoi à la différence entre eux, et lequel on préfère
#Aussi mettre la méthode, image, conclusion,...
#erreur couleur, 
#comparer a la solution existante
#pourquoi c'est la meillure
#qu'est ce qui est plus grave
#qu'elle conséquence pour chaque solution
# %%
ALD = {}
QLD = {}
GNB = {}
kppv = {}
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

# Convertir la colonne "Classement cardiologue" en label encodé
le = LabelEncoder()
df1['Code'] = le.fit_transform(df1['Classement cardiologue'])
df2['Code'] = le.fit_transform(df2['Classement cardiologue'])
df3['Code'] = le.fit_transform(df3['Classement cardiologue'])

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

counts = df1['Code'].value_counts().sort_index()

labels = counts.index.map({0: "Correcte", 1: "Artéfact"})

# Tracé
plt.figure(figsize=(6, 6))
plt.pie(counts.values, labels=labels, autopct='%1.1f%%', colors=['#d95f02', '#1b9e77'], startangle=90)
plt.title("Répartition des individus par classe")
plt.tight_layout()
plt.show()

#%%
C = pd.DataFrame.corr(df1.iloc[:,2:-1], method='pearson')
print(C)
plt.figure()
sns.heatmap(C, vmin=-1, vmax=1, cmap='coolwarm', annot=True)
plt.show()
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
# ADL
plt.figure()
adl = LinearDiscriminantAnalysis()
composantsald = adl.fit_transform(df1.iloc[:, 1:-1], df1["Code"])
Ra = np.random.randn(len(df1))

# Affichage
scatter = plt.scatter(composantsald[:, 0], Ra, c=df1["Code"], cmap='viridis')
plt.title("Représentation des données en fonction de Code")
plt.xlabel("Première composante ADL")
plt.ylabel("Valeur aléatoire (Ra)")

# Légende manuelle
import matplotlib.patches as mpatches
legend_labels = {0: "Artéfact", 1: "Correcte"}
colors = scatter.cmap(scatter.norm([0, 1]))  # Assure la cohérence des couleurs

patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(2)]
plt.legend(handles=patches, title="Classe")

plt.tight_layout()
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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import sem
import time

#arg
# @dataTrain sur lequel le jeu est entrainé
# @dataTest sur lequel le jeu est testé
# @type de classification qu'on veut faire
def train_predict(dataTrain,dataTest,func=False,display=False):
    y_pred = None
    precision = None
    if not func : #Si on est dans le cas des classement k neighbours
        X_train, X_test, y_train, y_test = train_test_split(dataTrain.iloc[:,2:-1] , dataTrain["Code"], test_size=1/3 , stratify = dataTrain["Code"])
        TotalIter = 100
        TrainingList = []
        adlList = {}
        maxItList = []
        forErrorMargin = {}
        for z in range(TotalIter): # On vient faire 100 courbes des n voisins
            precision = []
            X_t_train,X_val,y_t_train,y_val = train_test_split(X_train , y_train, test_size=1/3 , stratify = y_train)
            y_pred = []
            maxKneighb = 50#On sépare le jeu de données, et on dit qu'on va aller jusqu'à 50 voisins
            for i in range(1,maxKneighb+1):
                adl = KNeighborsClassifier(n_neighbors=i)
                dataTrained=adl.fit(X_t_train,y_t_train)
                y_pred.append(adl.predict(X_val))
                precision.append(adl.score(X_val, y_val))
                atI = adlList.get(i)
                if atI is None:
                    adlList[i] = [adl.score(X_val, y_val),adl]
                else:
                    if adl.score(X_val, y_val) > atI[0]:
                        adlList[i][0] = adl.score(X_val, y_val)
                        adlList[i][1] = adl

                   
            maxIt = precision.index(max(precision))+1
            maxItList.append(maxIt)
            
            TrainingList.append(precision)
            for index,precForThisNeighb in enumerate(precision):
                forErrorMargin[index] = forErrorMargin.get(index, []) + [precForThisNeighb]
            plt.plot(precision)

        #Précision moyenne c'est la moyenne de chaque case de traingin list donc la moyenne de chaque précision pour chaque nombre de voisins
        precisionMoyenne = []
        for j in range(maxKneighb):
            valeurs_j = [iteration[j] for iteration in TrainingList]
            precisionMoyenne.append(np.mean(valeurs_j))
        plt.show()
        plt.figure()

        x = []
        y_mean = []
        y_err = []
        
        for xi, values in forErrorMargin.items():
            x.append(xi)
            y_mean.append(np.mean(values))
            error = sem(values)  # erreur standard de la moyenne
            y_err.append(1.96 * error)  # intervalle de confiance à 95%
            print(intervalle_confiance_95(values))
        plt.errorbar(x, y_mean, yerr=y_err, fmt='o', label='Moyenne avec IC 95%')
        plt.grid(True)
        plt.plot(precisionMoyenne)
        plt.show()
        bestvoisin = max(set(maxItList), key=maxItList.count)
        print("Le meilleur est le {}".format(bestvoisin))

        bestAdl = adlList[int(np.median(maxItList))][1]
        scoreList = []
        X_dataTest = dataTest.iloc[:, 2:-1]
        y_dataTest = dataTest["Code"]
        for i in range(0,100):
            X_train, X_test, y_train, y_test = train_test_split(dataTrain.iloc[:,2:-1] , dataTrain["Code"], test_size=1/3 , stratify = dataTrain["Code"])
            kbest = KNeighborsClassifier(n_neighbors=bestvoisin)
            kbest.fit(X_train,y_train)
            precision_dataTest = kbest.score(X_dataTest, y_dataTest)
            scoreList.append(precision_dataTest)
            y_pred_final = kbest.predict(X_test)
            print(kbest.score(X_dataTest, y_dataTest))
            if display:
                print("Précision sur la prédiction sur le modèle de test")
                conf = confusion_matrix(y_dataTest, y_pred_final,normalize="true")
                disp = ConfusionMatrixDisplay(confusion_matrix=conf)
                disp.plot()
                plt.show()  
        print("Score : {} +- {}".format(np.mean(scoreList), intervalle_confiance_95(scoreList)))
    
    else :
        scoreList = []

        for i in range(0,100):
            X_train, X_test, y_train, y_test = train_test_split(dataTrain.iloc[:,2:-1] , dataTrain["Code"], test_size=1/3 , stratify = dataTrain["Code"])
            adl = func()
            dataTrained = adl.fit(X_train,y_train)
            #Prediction sur l'échantillon de test
            y_pred = adl.predict(X_test)
            #La réponse du train est dans y_test
            #Le résultat du pred est dans y_pred
            precision = adl.score(X_test, y_test)

            #Bien mais pas très détaillé
            if display:
                print("Précision de {}% - (entrainement)".format(100*sum(y_pred == y_test)/len(y_test)))
                conf = confusion_matrix(y_test, y_pred,normalize="true")
                disp = ConfusionMatrixDisplay(confusion_matrix=conf)
                disp.plot()
                plt.show()
                print("Répartition d'apprentissage : {}.\n".format(y_train.value_counts()))
                print("Répartition des tests : {}.\n".format(y_test.value_counts()))
        
            #On est critique sur le nombre d'individus
            
            
            
            
            
            
            #X c'est les données, et y la valeur que l'on veut prédire
            X_dataTest = dataTest.iloc[:, 2:-1]
            y_dataTest = dataTest["Code"]
            
            # Prédiction sur dataTest
            start_time = time.time()
            y_pred_dataTest = adl.predict(X_dataTest)
            end_time = time.time()
            #print(f"Prédiction sur dataTest a mis {end_time - start_time:.2f} secondes à s'exécuter.")
            # Évaluation
            precision_dataTest = adl.score(X_dataTest, y_dataTest)
            #print("Précision sur dataTest : {:.2f}%".format(100 * precision_dataTest))
            scoreList.append(precision_dataTest)
            
            # Matrice de confusion
            if display:
                conf_dataTest = confusion_matrix(y_dataTest, y_pred_dataTest, normalize="true")
                disp_dataTest = ConfusionMatrixDisplay(confusion_matrix=conf_dataTest)
                disp_dataTest.plot()
                plt.show()
        print("Score : {} +- {}".format(np.mean(scoreList), intervalle_confiance_95(scoreList)))
    return np.mean(scoreList), intervalle_confiance_95(scoreList)
    
    
        
    
#%%

















#Pour la présentation une slide rapide pour présenter les données
#plus détaillé sur la méthode , quoi en approtissage, etc
#présentation des résultats (en donnant tout les détails, quelle méthode donne quoi)
#qu'est ce qui est fait acutellement (voir matrice de confusion) et ce qu'on pourrait faire pour le remplacer
#présenter le jeu de données, 
#7 8 slides en gros
#Voir consignes sur chamilo




ALD["Intra"] = train_predict(df1, df2,LinearDiscriminantAnalysis)
ALD["Inter"] = train_predict(df1, df3,LinearDiscriminantAnalysis)
QLD["Intra"] = train_predict(df1, df2,QuadraticDiscriminantAnalysis)
QLD["Inter"] = train_predict(df1, df3,QuadraticDiscriminantAnalysis)
GNB["Intra"] = train_predict(df1, df2,GaussianNB)
GNB["Inter"] = train_predict(df1, df3,GaussianNB)
kppv["Intra"] = train_predict(df1, df2)
kppv["Inter"] = train_predict(df1, df3)


# Diagramme pour les valeurs "Inter"
methods = ['ALD', 'QLD', 'GNB', 'kppv']
means = [ALD["Inter"][0], QLD["Inter"][0], GNB["Inter"][0], kppv["Inter"][0]]
errors = [
    (ALD["Inter"][1][1] - ALD["Inter"][0]),
    (QLD["Inter"][1][1] - QLD["Inter"][0]),
    (GNB["Inter"][1][1] - GNB["Inter"][0]),
    (kppv["Inter"][1][1] - kppv["Inter"][0])
]

plt.figure(figsize=(8, 6))
plt.bar(methods, means, yerr=errors, capsize=5, color=['blue', 'orange', 'green', 'red'])
plt.ylabel('Score moyen')
plt.ylim(0.650, 0.775)  # Réduire l'échelle de l'axe y
plt.title('Performance des méthodes (Inter)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Diagramme pour les valeurs "Intra"
means = [ALD["Intra"][0], QLD["Intra"][0], GNB["Intra"][0], kppv["Intra"][0]]
errors = [
    (ALD["Intra"][1][1] - ALD["Intra"][0]),
    (QLD["Intra"][1][1] - QLD["Intra"][0]),
    (GNB["Intra"][1][1] - GNB["Intra"][0]),
    (kppv["Intra"][1][1] - kppv["Intra"][0])
]

plt.figure(figsize=(8, 6))
plt.bar(methods, means, yerr=errors, capsize=5, color=['blue', 'orange', 'green', 'red'])
plt.ylabel('Score moyen')
plt.ylim(0.7, 0.88)  # Réduire l'échelle de l'axe y
plt.title('Performance des méthodes (Intra)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()