from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

#chargement du dataset
data = pd.read_csv("Tetuan-PC.csv")

# Liste des prédicteurs disponibles
predictors = ['Temperature', 'Humidity', 'WindSpeed', 'PCZone2', 'PCZone3']

# Initialiser les variables pour le modèle optimal
best_adjusted_r2 = -float("inf")
best_model = None
best_combination = None

# Fonction pour calculer l'Adjusted R²
def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Tester toutes les combinaisons de 1 à 5 prédicteurs
n = len(data)  # Nombre d'observations
for k in range(1, len(predictors) + 1):
    for combination in combinations(predictors, k):
        # Sous-ensemble de prédicteurs
        X_subset = data[list(combination)]
        Y = data['PCZone1']
        
        # Ajuster le modèle
        model = LinearRegression()
        model.fit(X_subset, Y)
        
        # Calculer R² et Adjusted R²
        r2 = model.score(X_subset, Y)
        adj_r2 = adjusted_r2(r2, n, k)
        
        # Mettre à jour le meilleur modèle
        if adj_r2 > best_adjusted_r2:
            best_adjusted_r2 = adj_r2
            best_model = model
            best_combination = combination

print(f"Meilleur modèle : {best_combination}")
print(f"Adjusted R² : {best_adjusted_r2}")

# Extraire les coefficients et R² pour le meilleur modèle
coefficients = best_model.coef_
intercept = best_model.intercept_
r2 = best_model.score(data[list(best_combination)], data['PCZone1'])

# Afficher les résultats
print(f"Coefficients : {coefficients}")
print(f"Intercept : {intercept}")
print(f"R² : {r2}")

import numpy as np
from scipy.stats import t

# Calcul des p-valeurs pour chaque coefficient
n = len(data)
k = len(best_combination)
residuals = data['PCZone1'] - best_model.predict(data[list(best_combination)])
s_squared = np.sum(residuals**2) / (n - k - 1)
se_coefficients = np.sqrt(s_squared / (np.sum((data[list(best_combination)] - np.mean(data[list(best_combination)]))**2, axis=0)))

t_stats = coefficients / se_coefficients
p_values = [2 * (1 - t.cdf(abs(t_stat), df=n - k - 1)) for t_stat in t_stats]

print("P-valeurs pour les coefficients :")
for var, p in zip(best_combination, p_values):
    print(f"{var}: {p}")


# Nouvelles conditions
new_data = pd.DataFrame({
    'Temperature': [26],
    'Humidity': [65],
    'WindSpeed': [4.2],
    'PCZone2': [18840],
    'PCZone3': [25700]
})

# Garder seulement les colonnes utilisées dans le modèle
new_data = new_data[list(best_combination)]

# Faire la prédiction
prediction = best_model.predict(new_data)
print(f"Prédiction pour PCZone1 : {prediction[0]}")
