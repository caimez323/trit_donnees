
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

#   1.1 On charge les données
datas = pd.read_excel("ozone.xlsx")


#   1.2 On describe
desc = datas.describe()
#   1.2 On regarde le type de valeurs, ici tout va bien 
for d in datas:
    break
    print(datas[d].dtype)
#   1.3 vent_code doit être une catégorie pas un entier
datas["vent_code"] = datas["vent"].astype("object")
#   1.4 On supprime les lignes qui sont vide ou erreurs on ne va pas les considérer 
datas = datas.dropna()


#   2)
#   2.2
X_quant = datas.select_dtypes("number")
#   2.3
X_quant_mean = np.mean(X_quant)
X_quant_var = np.var(X_quant)

#   2.4
plt.figure()
X_quant.boxplot()
plt.show()

#   2.5
plt.figure()
X_quant.hist()
plt.show()

#   3)
#   3.1 on affiche la modalité
for column in X_quant.columns:
    break
    print(f"Variable: {column}")
    print(f"Nombre de modalités: {X_quant[column].nunique()}")
    print("Modalités et effectifs:")
    print(X_quant[column].value_counts())
    print("\n")


for element in X_quant:
    break
    (e, eff) = np.unique(X_quant[element], return_counts=True);
    plt.figure()
    plt.bar(e, eff, width = 0.5)
    plt.title('Répartition des observations en fonction du {}'.format(element))
    plt.xlabel('{}'.format(element))
    plt.ylabel('Nombre d\'observations')

#   4) analyse bivariées
#   4.1
plt.scatter(X_quant["T9"],X_quant["T12"])

#   4.2 commenté car trop lent
#pd.plotting.scatter_matrix(X_quant)

#   4.3
X_quant_corr = X_quant.corr()


#   5)
#   5.1

crosstab = pd.crosstab(datas['vent'], datas['pluie'])
#print(crosstab)
crosstab.plot(kind='bar', stacked=True)

#   6)
#On affiche les colonnes que dans x_quant
plt.figure()
for element in X_quant:
    datas.boxplot(column=element, by="pluie")
plt.show()



