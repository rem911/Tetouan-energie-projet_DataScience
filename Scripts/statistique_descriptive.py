#bibliothèque

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#chargement du dataset
data = pd.read_csv("Tetuan-PC.csv")

#affichage des premières lignes
print(data.head())

#résumé des dimensions
print("Nombre d'observations :", data.shape[0])
print("Nombre de variables :", data.shape[1])

#vérification des valeurs manquantes 
missing_values = data.isnull().sum()
print("Valeurs manquantes :\n", missing_values)

#résumé statistique
print(data.describe())

#boxplot_température
sns.boxplot(data=data, y="Temperature")
plt.title("Boxplot de la température")
plt.show()

#boxplot_consommation d'energie de zone1
sns.boxplot(data=data, y="PCZone1")
plt.title("Boxplot de la consommation énergétique de Zone 1")
plt.show()

#histogramme_distribution_humidité
data['Humidity'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution de l'humidité")
plt.xlabel("Humidité (%)")
plt.ylabel("Fréquence")
plt.show()

#scatter plots Température vs Consommation énergétique (Zone 1)
plt.scatter(data['Temperature'], data['PCZone1'], alpha=0.5, color='blue')
plt.title("Température vs Consommation énergétique (Zone 1)")
plt.xlabel("Température (°C)")
plt.ylabel("Consommation énergétique (Zone 1)")
plt.show()

#scatter plots Humidité vs Consommation énergétique (Zone 2)
plt.scatter(data['Humidity'], data['PCZone2'], alpha=0.5, color='green')
plt.title("Humidité vs Consommation énergétique (Zone 2)")
plt.xlabel("Humidité (%)")
plt.ylabel("Consommation énergétique (Zone 2)")
plt.show()

#Matrice de corrélation avec heatmap
numeric_data = data.drop(columns=['DateTime'])
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()

