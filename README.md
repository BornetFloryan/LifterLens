# Coach IA Powerlifting

Projet de coach IA de powerlifting combinant **machine learning prédictif** et **modèle de langage (LLM)** afin :
- d’estimer les performances en **Squat, Bench Press et Deadlift**,
- puis de générer automatiquement un **programme d’entraînement personnalisé**.

---

## Objectif du projet

Ce projet vise à démontrer la mise en œuvre complète d’un pipeline IA :

1. Exploitation de **données réelles à grande échelle**
2. Entraînement et comparaison de modèles de **régression supervisée**
3. Optimisation des hyperparamètres
4. Déploiement via une interface utilisateur (Streamlit)
5. Génération de recommandations à l’aide d’un LLM (Mistral)

---

## Données utilisées

### Source

Les données proviennent de la base **OpenPowerliftingg**, qui regroupe plusieurs millions de performances réelles issues de compétitions officielles.

🔗 Lien de téléchargement des données :
[Powerlifting Database - Kaggle](https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database)

Dans ce projet, le fichier utilisé est :
database/openpowerlifting-2024-01-06-4c732975.csv

Le code charge explicitement le fichier depuis le dossier database/ :

```
df = pd.read_csv(
    "database/openpowerlifting-2024-01-06-4c732975.csv",
    low_memory=False
)
```

---

### Prétraitement et filtrage

Les données ont été nettoyées afin de garantir leur qualité et leur cohérence :

- Compétitions **SBD uniquement** (Squat / Bench / Deadlift)
- Équipement **Raw**
- Suppression :
  - des catégories de sexe ambiguës (`MX`)
  - des valeurs manquantes
- Sélection volontaire d’un nombre réduit de variables pour garantir
  **simplicité, interprétabilité et généralisation**

**Variables d’entrée**
- Sexe
- Âge
- Poids de corps (kg)

**Variables cibles**
- Best3SquatKg
- Best3BenchKg
- Best3DeadliftKg

---

## Modélisation

### Problème traité
- Régression supervisée
- Prédiction continue des performances (kg)

### Modèles testés
- **XGBoost**
- **CatBoost**

Ces deux modèles appartiennent à la famille du **Gradient Boosting sur arbres de décision**, bien adaptée aux données tabulaires et aux relations non linéaires.

---

## Évaluation et optimisation

- Métrique utilisée : **MAE (Mean Absolute Error)**, exprimée en kilogrammes
- Validation croisée (3-fold)
- Recherche d’hyperparamètres via **RandomizedSearchCV**
- Analyse de la stabilité des performances pendant l’optimisation

Le modèle **XGBoost optimisé** est retenu pour son **meilleur compromis entre précision, stabilité et temps d’entraînement**.

---

## Déploiement

Une interface **Streamlit** permet à l’utilisateur de :
1. renseigner son sexe, âge, poids de corps et le nombre de jours d’entraînement,
2. obtenir une estimation de ses performances via le modèle ML,
3. recevoir un **programme d’entraînement personnalisé** généré par le LLM **Mistral** à partir de ces estimations.

---

## Technologies utilisées

- Python
- Pandas, Scikit-learn
- XGBoost, CatBoost
- Streamlit
- API Mistral (LLM)

---

## Perspectives d’amélioration

- Ajout du niveau de compétition
- Prise en compte de l’historique d’entraînement
- Modélisation de la progression à long terme
- Personnalisation avancée du volume et de l’intensité

---

## Auteur

## Auteur

[Floryan BORNET](https://github.com/BornetFloryan) 
[Corentin BRENDLÉ](https://github.com/BrendleCorentin)

