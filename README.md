# Projet OpenClassrooms - Home Credit Default Risk
## Ce projet réalisé dans le cadre du parcours OpenClassrooms porte sur l'analyse du jeu de données Home Credit Default Risk et inclut la création d'un modèle de prédiction.

## Contenu du Projet
### Tests
Le dossier Tests contient les tests réalisés à chaque push sur la branche main.

### model
Le dossier model héberge les fichiers nécessaires au déploiement du modèle de ML. Il s'agit un LGBMClassifier entraîné sur les données d'entrainement.

### templates
Le répertoire templates contient les fichiers html pour le rendu de l'app.

### fichiers
app.py : contient l'app utilisée via heroku.
fonction.py : contient la fonction testée.
noteboo.ipynb : contient l'exploration et la recherche de modèle pour le projet.
Procfile : fichier nécessaire pour le déploiement avec Heroku


## Utilisation
### Notebooks
Pour explorer les analyses et les résultats de la modélisation, ouvrez le fichier .ipynb dans Jupyter Notebook ou JupyterLab.

### API
L'API est déployée automatiquement sur Heroku à l'adresse suivante : 
https://home-credi-default-risk-2d75b983d33b.herokuapp.com/

Le format du fichier nécessaire est le fichier complet présent dans le notebook.

l'application streamlit-app permet de faire une simulation de cette api avec des données pré-enregistrées.

# Auteur
Ce projet a été réalisé par Heneault Thomas
