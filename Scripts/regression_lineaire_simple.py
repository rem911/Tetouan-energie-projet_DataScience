from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd

#chargement du dataset
data = pd.read_csv("Tetuan-PC.csv")

# Exclure les colonnes non numériques
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# corrélations avec PCZone1
correlations = numeric_data.corr()['PCZone1'].sort_values(ascending=False)
print("Corrélations avec PCZone1 :\n", correlations)

# Définir X (PCZone2) et Y (PCZone1)
X = data[['PCZone2']]
Y = data['PCZone1']

# modèle
model = LinearRegression()
model.fit(X, Y)

# Obtenir les coefficients
beta_0 = model.intercept_
beta_1 = model.coef_[0]
print(f"Coefficient (β0) : {beta_0}")
print(f"Coefficient (β1) : {beta_1}")

n = len(X) 
df = n - 2 

# Erreur standard de β1
# Calculer les prédictions du modèle
Y_pred = model.predict(X)
residuals = Y - Y_pred
s_squared = np.sum(residuals**2) / df  # Variance des résidus
X_std = (X - np.mean(X))**2
SE_beta1 = np.sqrt(s_squared / np.sum(X_std))

# Valeur critique t pour un intervalle à 95%
t_crit = stats.t.ppf(1 - 0.025, df)

# Calcul de l'intervalle de confiance
lower_bound = beta_1 - t_crit * SE_beta1
upper_bound = beta_1 + t_crit * SE_beta1
print(f"Intervalle de confiance à 95% pour β1 : [{lower_bound}, {upper_bound}]")

from scipy.stats import t

# Calculer la statistique t
t_stat = beta_1 / SE_beta1

# Calculer les degrés de liberté
n = len(X)
df = n - 2  # Degrés de liberté pour une régression simple

# Calculer la p-valeur
p_value = 2 * (1 - t.cdf(abs(t_stat), df))

# Afficher les résultats
print(f"Statistique t : {t_stat}")
print(f"p-valeur : {p_value}")


from sklearn.metrics import r2_score

# Calcul_R²
r2 = r2_score(Y, Y_pred)
print(f"Coefficient de détermination (R²) : {r2}")
