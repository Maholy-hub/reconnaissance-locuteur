# Reconnaissance du locuteur

Réseau de neurones (CNN sur MFCC) qui identifie **qui parle** parmi 3 classes :
`moi`, `personne_choisie`, `autre`. Le verrou s'ouvre pour les deux premières classes
et reste fermé pour `autre`.

## Dataset
Dataset public **LibriSpeech (dev-clean)**, téléchargé automatiquement par le notebook
(il n'est pas dans ce dépôt).

- `moi` : locuteur 2277 du dataset
- `personne_choisie` : locuteur 2035 du dataset
- `autre` : les 38 autres locuteurs

Séparation train / validation / test **sans fuite de données** : par fichier pour les deux
locuteurs ciblés, par locuteur pour `autre` (le test contient des voix jamais entendues).

## Traitement du signal
Audio 16 kHz mono, suppression des silences, segments de 2 s, 40 MFCC normalisés par segment.

## Modèle
CNN : 3 blocs Conv2D + BatchNormalization, GlobalAveragePooling, Dropout, Dense(3, softmax).
Poids de classes pour le déséquilibre, early stopping.

## Résultats
- Test par segment de 2 s : accuracy **87.64 %** (voir `resultats/`)
- Test par fichier entier, seuil de confiance 0.70 : faux acceptés 5,0 %, faux rejetés 17,9 %,
  aucune confusion entre `moi` et `personne_choisie`

## Utilisation
    pip install -r requirements.txt
    python predict.py audio.wav
    python predict.py audio.wav --seuil 0.70

## Limites
- Seulement 2 locuteurs ciblés et 28 fichiers de test pour ces classes : les pourcentages
  par classe cible sont peu précis.
- 5 % de faux acceptés : insuffisant pour un vrai système de sécurité.
- Le modèle reconnaît les locuteurs d'entraînement ; il faudrait réentraîner pour une autre personne.
