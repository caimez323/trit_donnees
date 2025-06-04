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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

img_data = pd.read_excel("database.xlsx",index_col=0)
img_data = img_data.dropna()

print(img_data)

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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

# 1. Données
X = img_data_reduit
y = img_data.iloc[:, -1]

# 2. Split train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# 3. Standardisation
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Grille d’hyperparamètres
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,), (150,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu']
}

# 5. GridSearchCV
mlp = MLPClassifier(max_iter=500, random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# 6. Résultats
print("Meilleurs paramètres :", grid_search.best_params_)
print("Score sur test set :", grid_search.score(X_test_scaled, y_test))

# 7. Prédictions + rapport
y_pred = grid_search.predict(X_test_scaled)
print("\nRapport de classification :\n", classification_report(y_test, y_pred))

#%%
from sklearn.svm import SVC
best_score_clf_lin = 0
best_confusion_matrix_clf_lin = None
scorelist_clf_lin = []
# Test de SVC avec différentes valeurs de C
for i in range(1, 10):
    print("Iteration {}".format(i)) 
    # On divise les données en train et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
    clf = SVC(kernel='linear',C=1)#Test avec linéaire, rbf pourrait être bien aussi à tester
    #on peut augementer légèrement la puiissance du modèle en augmentant C
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    scorelist_clf_lin.append(score)
    confusionMatrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
    if score > best_score_clf_lin:
        best_score_clf_lin = score
        best_confusion_matrix_clf_lin = confusionMatrix
disp_dataTest = ConfusionMatrixDisplay(confusion_matrix=best_confusion_matrix_clf_lin)
disp_dataTest.plot()
print("Score CLF Linéaire: {} +- {}".format(np.mean(scorelist_clf_lin), intervalle_confiance_95(scorelist_clf_lin)))


# %%
best_score_clf = 0
best_confusion_matrix_clf = None
scorelist_clf = []
# Test de SVC avec différentes valeurs de C
for i in range(1, 10):
    print("Iteration {}".format(i))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
    clf = SVC(C=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    scorelist_clf.append(score)
    confusionMatrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
    if score > best_score_clf:
        best_score_clf = score
        best_confusion_matrix_clf = confusionMatrix
disp_dataTest = ConfusionMatrixDisplay(confusion_matrix=best_confusion_matrix_clf)
disp_dataTest.plot()
print("Score CLF : {} +- {}".format(np.mean(scorelist_clf), intervalle_confiance_95(scorelist_clf)))
# %%

from sklearn.neural_network import MLPClassifier
best_score_MLP = 0
best_confusion_matrix_MLP = None
scorelist_MLP = []
for i in range(1,10):
    print("Iteration {}".format(i)) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3)
    clf = MLPClassifier(hidden_layer_sizes=(i), max_iter=5000,alpha=0.01)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    scorelist_MLP.append(score)
    confusionMatrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
    if score > best_score_clf:
        best_score_clf = score
        best_confusion_matrix_MLP= confusionMatrix
disp_dataTest = ConfusionMatrixDisplay(confusion_matrix=best_confusion_matrix_MLP)
disp_dataTest.plot()
print("Score MLP : {} +- {}".format(np.mean(scorelist_MLP), intervalle_confiance_95(scorelist_MLP)))

# %%
# Affichage des scores moyens et intervalles de confiance à 95% pour chaque classifieur

# Moyennes et intervalles
scores = [
    ("SVC Linéaire", np.mean(scorelist_clf_lin), intervalle_confiance_95(scorelist_clf_lin)),
    ("SVC (C=100)", np.mean(scorelist_clf), intervalle_confiance_95(scorelist_clf)),
    ("MLP", np.mean(scorelist_MLP), intervalle_confiance_95(scorelist_MLP))
]

labels = [s[0] for s in scores]
moyennes = [s[1] for s in scores]
erreurs = [max(abs(s[1] - s[2][0]), abs(s[2][1] - s[1])) for s in scores]  # demi-largeur de l'IC

plt.figure(figsize=(8, 6))
colors = ['skyblue', 'salmon', 'limegreen']
for i, (label, moyenne, erreur) in enumerate(zip(labels, moyennes, erreurs)):
    plt.errorbar(label, moyenne, yerr=erreur, capsize=10, color=colors[i], fmt='o', linestyle='None', label=label)
plt.ylabel("Score moyen (accuracy)")
plt.ylim(0.8, 1.0)
plt.title("Scores moyens des classifieurs avec intervalle de confiance à 95%")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# %%
