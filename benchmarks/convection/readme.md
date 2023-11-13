# Benchmark - Convection Slab (2023)

Le use-case principal du benchmark est un run de convection en slab. Le run permet d’évaluer trois noyaux de calcul : l’update hydro, la conduction thermique et la viscosité. Il nécessite l’utilisation de 5 niveaux d’AMR. Le run d’évaluation consiste en 100 itérations des noyaux de calcul, 100 cycles AMR et 2 cycle de load-balancing. En dehors du restart nécessaire à l’initialisation, aucun IO ne sera effectué.

## Préparation des runs de convection

(voir `generate_restart/readme.md`)

**Le résultat du run préliminaire (`generate_restart/3_refine_2/restart_C91_100.h5`) peut (et doit) être réutilisé entre les runs et entre les architectures**

## Runs d’évaluation des performances

Enfin, l’évaluation est effectuée sur un run tilé à partir du dernier restart raffiné. Le tiling utilise la condition initiale tiled_restart et le nombre de processus MPI correspond au nombre de réplications totales. Puisque le run de convection est un run stratifié sur la direction z, on ne tile que sur les directions x et y. 

![bench](uploads/281eaca7ea75d3f22b6c614db2166387/bench.png)

L’adaptation à un nouveau tiling doit être fait dans les parties du .ini suivantes : 
`restart/nrep_x` et `restart/nrep_y` doivent indiquer le nombre de réplications totales
`amr/level_min` et `amr/level_max` doivent refléter le nouveau maillage.
`amr/coarse_oct_resolution_x` et `amr/coarse_oct_resolution_y` doivent reproduire le slab correspondant au nouveau `amr/level_min`
`mesh/xmax` et `mesh/ymax` peuvent être agrandies aussi même si ce n’est pas absolument nécessaire.

**Le script `run_scripts/bench.py` lance des jobs avec sbatch, lisez le avant de l'executer**

Le script python `run_scripts/bench.py` a été utilisé pour effectuer les mesures sur différentes machines et architectures, vous pouvez le modifier pour ajouter des machines et des architectures. Vous pouvez le lancer depuis le dossier `benchmarks/convection` apres avoir crée un lien symbolique vers l'executable pour l'architecture cible et après avoir copié le fichier de restart (`generate_restart/3_refine_2/restart_C91_100.h5`, ou téléchargé depuis une autre source) sous le nom `restart.h5` dans le dossier `benchmarks/convection`. 

le dossier `run_scripts` contient les templates pour les jobs sur les différentes machines, le script `run_scripts/bench.py` remplace les placeholders dans les templates marqués par `<>` pour créer des dossier pour le benchmark de weak scaling et lancer les jobs. 

