#bibliotheques
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


#chargement du dataset
data = pd.read_csv("Tetuan-PC.csv")

#filtrage_variable
features = ['Temperature', 'Humidity', 'WindSpeed', 'PCZone1', 'PCZone2', 'PCZone3']
data_pca = data[features]

#calcul_variance
print("Variance de chaque variable :\n", data_pca.var())

#standardisation
scaler = StandardScaler()
data_pca_scaled = scaler.fit_transform(data_pca)

# Application_PCA_sur_données_standardisées
pca = PCA()
pca_components = pca.fit_transform(data_pca_scaled)

# composantes_principales
print("Composantes principales (loading vectors) :\n", pca.components_)

#graphe_loading_vectors
loading_df = pd.DataFrame(pca.components_[:2], columns=features, index=['PC1', 'PC2'])

# Visualisation
loading_df.T.plot(kind='bar', figsize=(10, 6))
plt.title("Contributions des variables aux 2 premières composantes principales")
plt.ylabel("Charge (loading)")
plt.xlabel("Variables")
plt.grid()
plt.show()

# Pourcentage de variance expliquée par chaque composante
pve = pca.explained_variance_ratio_

# Variance expliquée cumulée
cumulative_pve = pve.cumsum()

# Affichage des résultats
print("Pourcentage de variance expliquée par chaque composante :\n", pve)
print("Pourcentage de variance expliquée cumulée :\n", cumulative_pve)
import matplotlib.pyplot as plt

# Tracer le PVE et le PVE cumulatif
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pve)+1), pve, alpha=0.5, align='center', label='PVE individuel')
plt.step(range(1, len(cumulative_pve)+1), cumulative_pve, where='mid', label='PVE cumulatif')
plt.xlabel('Composantes principales')
plt.ylabel('Pourcentage de variance expliquée')
plt.title('Pourcentage de Variance Expliquée (PVE) par les Composantes Principales')
plt.legend(loc='best')
plt.grid()
plt.show()

def plot_correlation_circle(pca, components, feature_names):
    plt.figure(figsize=(8, 8))

    # Tracer les flèches pour chaque variable
    for i, (x, y) in enumerate(zip(components[0], components[1])):
        plt.arrow(0, 0, x, y, color='r', alpha=0.5, head_width=0.02)
        plt.text(x, y, feature_names[i], fontsize=12, ha='center', va='center')

    # Ajouter le cercle unitaire
    circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--', alpha=0.7)
    plt.gca().add_artist(circle)

    # Configurer l'axe
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel("Composante principale 1 (PC1)")
    plt.ylabel("Composante principale 2 (PC2)")
    plt.title("Cercle de corrélation (PC1 vs PC2)")
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Appeler la fonction pour tracer le cercle de corrélation
features = ['Temperature', 'Humidity', 'WindSpeed', 'PCZone1', 'PCZone2', 'PCZone3']
plot_correlation_circle(pca, pca.components_[:2], features)