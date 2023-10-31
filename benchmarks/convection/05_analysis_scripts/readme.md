# Scripts d'analyse pour le benchmark

Deux scripts principaux : analyze.py et verification.py

## `verification.py`

Permet de vérifier si les runs sont valides.
Le script va chercher dans des sous-dossiers par "type de run". Les dossiers inspectés sont : `JZ_v100`, `JZ_a100`, `JZ_csl`, `AA_genoa` et `AA_mi250`. Le script doit donc être placé ainsi que l'utilitaire `dump_logs.sh` à la racine des runs (là où le script `bench.py` a été lancé).
Pour chaque dossier, le script efface le dossier `logs` s'il existe, puis le recréé et le remplit en utilisant `dump_logs.sh`. Le dossier regroupe l'ensemble des logs de chaque run par catégorie (JZ_v100, etc.).
Enfin, le script lit les outputs et effectue un rapport sur chaque run.

Les runs qui terminent sans souci sont ajoutés à un plot dt/temps tandis que les runs qui présentent des densités négatives, des pressions négatives ou qui se sont terminés sur une exception seront ajoutés à une liste de runs invalides.
Les plots sont à vérifier manuellement. Le dt devrait être quasi constant sur ces runs.

## `analyze.py`

Script permettant d'extraire les timers d'un run. Il prend en 1er paramètre un nom de fichier de log. Il est possible de spécifier les options suivantes :

`--by-time` -> Trie les sorties par temps d'exécution du noyau descendant plutôt qu'alphabétique
`--gdoc-format` -> Sort un résultat sans légende avec les colonnes suivantes séparées d'un espace :
  0: Temps total indiqué sur CPU à la fin du run
  1: Temps total indiqué dans les timers
  2: AMR
  3: Cooling
  4: Hydro
  5: MPI ghosts
  6: Parabolic/thermal conduction
  7: Parabolic/viscosity
  8: Calcul du DT


