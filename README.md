# Projet_final

**Ce projet à pour but de déterminer le risque qu’un arbre allergisant soit planté ou déjà présent dans une zone pour la faire éviter par les personnes allergiques.
Le dataset provient du site web https://opendata.paris.fr/explore/dataset/les-arbres. Il s'agit d'une base de données de 204727 lignes et de 17 colonnes. Cette base est nécessaire pour le projet parce qu'elle contient toutes les informations requises pour cette étude en grande quantité.**

**Pour cette étude, de la base de données ont été utilisés les colonnes  ARRONDISSEMENT, GENRE, LIEU/ADRESSE, STADE, Allergisant, ALLERGIE, LATITUDE et LONGITUDE.**

**Comme il ne manquait des valeurs que pour la LATITUDE et la LONGITUDE sur une seule de ligne, cette ligne a été supprimée.**

**Les colonnes utilisées pour le dataset sont GENRE, ALLERGIE, LATITUDE et LONGITUDE.**

**Le critère de sélection des lignes de cette base est que la colonne ALLERGIE doit valoir 1. Ce qui signifie que l'arbre cause une allergie.**

**Les modèles candidats pour la prédiction sont le SVD, le NMF et enfin le Co Clustering. Le modèle retenu étant le SVD.**

**Pour géolocaliser, l'utilisation de la librairie Folium était recommandé.**

**Performances : RMSE 1.1253, MAE 0.6335 du modèle SVD pour la prédiction.**
