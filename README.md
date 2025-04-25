# Analyse de la consommation énergétique à Tétouan

Ce projet a été réalisé dans le cadre du module de Data Science à l’ISEP. Il porte sur l’analyse statistique d’un jeu de données réel issu de la ville de Tétouan (Maroc), comprenant des variables climatiques et énergétiques. L’objectif principal est de comprendre les relations entre ces variables et de construire un modèle prédictif de la consommation électrique à partir d’indicateurs climatiques et des consommations dans d’autres zones.

## Objectifs du projet

- Réaliser une analyse descriptive du jeu de données
- Appliquer une Analyse en Composantes Principales (ACP) pour réduire la dimension
- Construire un modèle de régression linéaire simple et multiple
- Prédire la consommation énergétique de la Zone 1 à partir de variables pertinentes

## Données utilisées

Le jeu de données comprend 13 248 observations réparties sur plusieurs variables :
- Variables climatiques : Température, Humidité, Vitesse du vent
- Variables énergétiques : Consommation des zones PCZone1, PCZone2, PCZone3

## Méthodologie

1. **Prétraitement et nettoyage des données** : inspection des types, gestion des valeurs manquantes.
2. **Analyse descriptive** : statistiques de base, visualisations, matrice de corrélation.
3. **Analyse en Composantes Principales (ACP)** : exploration de la structure des données.
4. **Régression linéaire simple** : entre PCZone1 et PCZone2.
5. **Régression linéaire multiple** : prédiction de PCZone1 à partir de 5 variables.
6. **Prédiction finale** : calcul de la consommation attendue selon des conditions données.

## Résultat de prédiction

Pour les conditions suivantes :
- Température : 26 °C
- Humidité : 65 %
- Vitesse du vent : 4.2 km/h
- Consommation Zone 2 : 18 840 KW
- Consommation Zone 3 : 25 700 KW

Le modèle prédit une consommation pour la Zone 1 de **29,802.29 KW**.

## Arborescence du projet

```
.
├── data/                    # Jeu de données brut
├── scripts/                 # Scripts Python pour analyse descriptive, PCA, régressions linéaires
├── notebooks/               # Notebook Jupyter
├── figures/                 # Graphiques générés
├── rapport/                 # Rapport final (PDF)
├── requirements.txt         # Librairies nécessaires
└── README.md

```
## Technologies utilisées

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- Jupyter Notebook

## Auteure

Ivan remy Simo Mendje 
Projet réalisé dans le cadre du module Data Science – ISEP – 2024/2025.
