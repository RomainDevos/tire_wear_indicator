# Organisation des fichiers

Ce document decrit comment est organise le repertoire `tire_wear_indicator` et comment retrouver les differents notebooks et diagrammes utilises pour le projet de detection d'usure des pneus.

## Structure generale

```
tire_wear_indicator/
`-- notebooks_kaggle/
    |-- Code_BASE/
    |   |-- mobilenetv3.ipynb
    |   |-- projet-michelin.ipynb
    |   `-- Organisation-dataset/
    |       |-- Organisation_fichier_mobilenet.png
    |       `-- Organisation_fichier_projet_michelin.png
    `-- Code_Data_Aug/
        |-- Codes/
        |   |-- projet-michelin-augmented-base.ipynb
        |   `-- projet-michelin-augmented-yolo.ipynb
        `-- Organisation/
            |-- Organisation_dataset_aug_base.png
            `-- Organisation_yolo.png
```

## Dossiers et fichiers clefs

### notebooks_kaggle
Contient tous les notebooks Jupyter et documents de reference exportes depuis Kaggle.

### Code_BASE
Cadre de travail pour les experiences de base.
- `mobilenetv3.ipynb` explore un modele MobileNetV3 sur le dataset original.
- `projet-michelin.ipynb` sert de notebook principal pour les essais sans augmentation.
- Le sous-dossier `Organisation-dataset/` stocke des schemas explicatifs (.png) montrant comment sont ranges les fichiers sources pour chacun des notebooks de base.

### Code_Data_Aug
Travail dedie aux scripts avec augmentation de donnees.
- Le sous-dossier `Codes/` regroupe les deux notebooks experimentaux (version base et version YOLO) avec les transformations d'augmentation.
- Le sous-dossier `Organisation/` contient les schemas (.png) qui documentent la structure attendue des datasets et sorties pour chaque pipeline d'augmentation.

## Comment utiliser cette organisation
1. Choisir un notebook dans `Code_BASE` ou `Code_Data_Aug` selon que l'on souhaite lancer une experience de reference ou avec augmentation.
2. Lire le schema correspondant dans `Organisation-dataset/` ou `Organisation/` pour mettre en place les repertoires de donnees locaux avant d'executer le notebook.


3. Préparer les datasets au bon format :

Pour les expériences de base (Code_BASE) : utiliser les datasets fournis par le professeur.

Pour les notebooks avec augmentation (Code_Data_Aug) :

- Dataset augmenté pour augmented-base :
https://kaggle.com/datasets/219254f69504eb9a9c233a4e835a3c67e2468b47cf8c536a8d588e993a53f3c0

- Dataset déjà formaté pour YOLO :
https://kaggle.com/datasets/e0e393c8b1ef18df2fa55bc80699a64cafe71006882ab30d77b0d717e4e34451

4. Ajouter tout nouveau notebook dans le dossier approprie et ranger ses diagrammes d'organisation a cote pour garder une documentation a jour.
