# Benchmark - Convection Slab (2023)

Le use-case principal du benchmark est un run de convection en slab. Le run permet d’évaluer trois noyaux de calcul : l’update hydro, la conduction thermique et la viscosité. Il nécessite l’utilisation de 5 niveaux d’AMR. Le run d’évaluation consiste en 100 itérations des noyaux de calcul, 100 cycles AMR et 2 cycle de load-balancing. En dehors du restart nécessaire à l’initialisation, aucun IO ne sera effectué.

## Préparation des runs de convection

(voir `generate_restart/readme.md`)

**Le résultat du run préliminaire (`generate_restart/3_refine_2/restart_C91_100.h5`) peut (et doit) être réutilisé entre les runs et entre les architectures**

## Runs d’évaluation des performances

Enfin, l’évaluation est effectuée sur un run tilé à partir du dernier restart raffiné. Le tiling utilise la condition initiale tiled_restart et le nombre de processus MPI correspond au nombre de réplications totales. Puisque le run de convection est un run stratifié sur la direction z, on ne tile que sur les directions x et y. 

L’adaptation à un nouveau tiling doit être fait dans les parties du .ini suivantes : 
* `restart/nrep_x` et `restart/nrep_y` doivent indiquer le nombre de réplications totales
* `amr/level_min` et `amr/level_max` doivent refléter le nouveau maillage.
* `amr/coarse_oct_resolution_x` et `amr/coarse_oct_resolution_y` doivent reproduire le slab correspondant au nouveau `amr/level_min`
* `mesh/xmax` et `mesh/ymax` peuvent être agrandies aussi même si ce n’est pas absolument nécessaire.

**Le script `run_scripts/bench.py` lance des jobs avec sbatch, lisez le avant de l'executer**

Le script python `run_scripts/bench.py` a été utilisé pour effectuer les mesures sur différentes machines et architectures, vous pouvez le modifier pour ajouter des machines et des architectures. Vous pouvez le lancer depuis le dossier `benchmarks/convection` apres avoir crée un lien symbolique vers l'executable pour l'architecture cible et après avoir copié le fichier de restart (`generate_restart/3_refine_2/restart_C91_100.h5`, ou téléchargé depuis une autre source) sous le nom `restart.h5` dans le dossier `benchmarks/convection`. 

le dossier `run_scripts` contient les templates pour les jobs sur les différentes machines, le script `run_scripts/bench.py` remplace les placeholders dans les templates marqués par `<>` pour créer des dossier pour le benchmark de weak scaling et lancer les jobs. 

Etape par étape :

* Compiler le code pour l'architecture cible (voir readme de Dyablo et wiki "Build commands on specific systems"). Attention de bien utiliser les mêmes modules que dans les templates du dossier `run_scripts`

* Créer un lien symbolique pour l'executable (`build/dyablo/test/solver/test_solver`) sous la forme `dyablo_<target>` dans le dossier `benchmarks/convection` (voir `executable_path` pour votre cible dans `bench.py`. ex : dyablo_v100 pour Jean Zay V100)

* Vérifier les runs que vous voulez executer, les allocations et les comptes sur les machines dans `bench.py`

* lancer `python3 run_scripts/bench.py <target>`

Exemple des commandes pour Jean Zay V100 (depuis la racine de Dyablo) :
```
$ mkdir build_v100
$ cd build_v100
$ module load cmake/ gcc/8.5.0 cuda/12.1.0 openmpi/4.0.5-cuda hdf5/1.12.0-mpi-cuda
$ cmake -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH="VOLTA70" -DDYABLO_CMAKE_ARGS="-DCMAKE_EXE_LINKER_FLAGS=-lstdc++fs" ..
$ make -j5 
$ cd ../benchmarks/convection/
$ ln -s ../../build_v100/dyablo/test/solver/test_solver dyablo_v100
$ emacs run_scripts/bench.py 
<read and verify bench.py>
$ python3 run_scripts/bench.py JZ_v100
```

Les résultats se trouvent dans le dossier du même nom que le paramètre <target> donné à `bench.py`, il contient plusieurs runs avec un nombre de ressources allouées différentes. Chaque sous-dossier contient un fichier `log.out` avec les logs d'execution et un fichier `timers.txt` avec les différents timers mesurés (le timer Total/TimeLoop est celui qui représente le mieux la performance totale du run).

```
$ ls JZ_v100/
run_N1x1_R1x1  run_N1x2_R2x1  run_N1x4_R2x2  run_N2x4_R4x2  run_N4x4_R4x4  run_N8x4_R8x4
$ ls JZ_v100/run_N1x1_R1x1/
dyablo_v100  job_tmpl_JZ_v100.slurm  last.ini  log.out  restart.h5  restart.ini  timers.txt
$ tail -n 30 JZ_v100/run_N1x1_R1x1/log.out 
Rank 0: old morton interval [0, 18446744073709551615[
Rank 0: iOct interval [0, 7842[
Rank 0: actual morton interval [0, 916992]
LightOctree rehash ...
LoadBalance - rank 0 octs : 7842 (0) -> 7842 (0)
Final scalar_data : iter=100 dt=2.72478e-05 time=0.00272478 
Total                     time (CPU) :    37.769 s (100.00%) , (GPU) :    37.769 s (100.00%)
| Init                      time (CPU) :     1.561 s (  4.13%) , (GPU) :     1.561 s (  4.13%)
| | initial_conditions        time (CPU) :     1.481 s (  3.92%) , (GPU) :     1.482 s (  3.92%)
| | other                     time (CPU) :     0.079 s (  0.21%) , (GPU) :     0.079 s (  0.21%)
| TimeLoop                  time (CPU) :    36.208 s ( 95.87%) , (GPU) :    36.208 s ( 95.87%)
| | AMR                       time (CPU) :     4.417 s ( 11.69%) , (GPU) :     4.333 s ( 11.47%)
| | | AMR: Mark cells           time (CPU) :     2.139 s (  5.66%) , (GPU) :     2.139 s (  5.66%)
| | | AMR: adapt                time (CPU) :     0.223 s (  0.59%) , (GPU) :     0.223 s (  0.59%)
| | | AMR: remap userdata       time (CPU) :     1.933 s (  5.12%) , (GPU) :     1.934 s (  5.12%)
| | | MPI ghosts                time (CPU) :     0.089 s (  0.24%) , (GPU) :     0.006 s (  0.02%)
| | | other                     time (CPU) :     0.032 s (  0.08%) , (GPU) :     0.031 s (  0.08%)
| | AMR: load-balance         time (CPU) :     0.050 s (  0.13%) , (GPU) :     0.050 s (  0.13%)
| | Cooling FF                time (CPU) :     0.004 s (  0.01%) , (GPU) :     0.088 s (  0.23%)
| | HydroUpdate_euler         time (CPU) :    17.851 s ( 47.26%) , (GPU) :    17.851 s ( 47.26%)
| | MPI ghosts                time (CPU) :     0.007 s (  0.02%) , (GPU) :     0.007 s (  0.02%)
| | Parabolic[explicit] thermal_conduction time (CPU) :     9.014 s ( 23.86%) , (GPU) :     9.038 s ( 23.93%)
| | Parabolic[explicit] viscosity time (CPU) :     3.781 s ( 10.01%) , (GPU) :     3.783 s ( 10.02%)
| | checkpoint                time (CPU) :     0.001 s (  0.00%) , (GPU) :     0.001 s (  0.00%)
| | dt                        time (CPU) :     0.342 s (  0.91%) , (GPU) :     0.342 s (  0.91%)
| | outputs                   time (CPU) :     0.001 s (  0.00%) , (GPU) :     0.001 s (  0.00%)
| | other                     time (CPU) :     0.742 s (  1.96%) , (GPU) :     0.714 s (  1.89%)
| checkpoint                time (CPU) :     0.000 s (  0.00%) , (GPU) :     0.000 s (  0.00%)
| outputs                   time (CPU) :     0.000 s (  0.00%) , (GPU) :     0.000 s (  0.00%)
| other                     time (CPU) :     0.000 s (  0.00%) , (GPU) :     0.000 s (  0.00%)
$ cat JZ_v100/run_N1x1_R1x1/timers.txt 
Rank ; Total ; Total/Init ; Total/Init/initial_conditions ; Total/TimeLoop ; Total/TimeLoop/AMR ; Total/TimeLoop/AMR/AMR: Mark cells ; Total/TimeLoop/AMR/AMR: adapt ; Total/TimeLoop/AMR/AMR: remap userdata ; Total/TimeLoop/AMR/MPI ghosts ; Total/TimeLoop/AMR: load-balance ; Total/TimeLoop/Cooling FF ; Total/TimeLoop/HydroUpdate_euler ; Total/TimeLoop/MPI ghosts ; Total/TimeLoop/Parabolic[explicit] thermal_conduction ; Total/TimeLoop/Parabolic[explicit] viscosity ; Total/TimeLoop/checkpoint ; Total/TimeLoop/dt ; Total/TimeLoop/outputs ; Total/checkpoint ; Total/outputs
0 ; 37.7691 ; 1.5608 ; 1.48142 ; 36.208 ; 4.41674 ; 2.13936 ; 0.223132 ; 1.93341 ; 0.0892187 ; 0.0498322 ; 0.00394448 ; 17.8508 ; 0.00675841 ; 9.01357 ; 3.78081 ; 0.000619446 ; 0.342294 ; 0.000798226 ; 2.20677e-06 ; 2.49734e-06
```


## Vérification des résultats

Les résultats sont vérifiés en analysant les logs pour verifier qu'il n'y a pas eu de crash et en tracant le `dt` au cours des itérations. Le script `verification/verification.py` analyse les logs générés par `bench.py` pour déterminer si les runs sont valides. Le script va chercher dans des sous-dossiers par "type de run". Les dossiers inspectés sont : `JZ_v100`, `JZ_a100`, `JZ_csl`, `AA_genoa` et `AA_mi250`. `verification/verification.py` doit être lancé depuis le dossier `benchmarks/convection`.
Pour chaque dossier, le script efface le dossier `logs` s'il existe, puis le recréé et le remplit en utilisant `dump_logs.sh`. Le dossier regroupe l'ensemble des logs de chaque run par catégorie (JZ_v100, etc.).
Enfin, le script lit les outputs et effectue un rapport sur chaque run.

Les runs qui terminent sans souci sont ajoutés à un plot dt/temps tandis que les runs qui présentent des densités négatives, des pressions négatives ou qui se sont terminés sur une exception seront ajoutés à une liste de runs invalides.
Les plots sont à vérifier manuellement. Le dt devrait être quasi constant sur ces runs.


## Analyse des performances

Les performances sont mesurées par les timers inclus dans le code. Un résumé facile à lire est écrit dans le fichier log.out, et un fichier timers.txt liste tous les timers en CSV pour chaque processus. 
Le timer qui représente le mieux le temps d'execution qui nous interesse est le timer `TimeLoop` qui représente le temps passé dans la boucle en temps. Certains sous-timers de `TimeLoop` nous interessent aussi pour savoir la proportion de temps passée dans chaque portion du code :
* AMR : Le temps passé dans le cycle de raffinement/déraffinement AMR
* MPI ghosts : les communications MPI (hors cycle AMR)
* HydroUpdate_euler : le noyau hydrodrynamique
* Parabolic[explicit] thermal_conduction/viscosity : les noyaux paraboliques

Le script `analyse/plot_times.py` génère un plot des différents runs lancés avec `bench.py`. Il doit être lancé depuis `benchmarks/convection` et prend en paramètre un dossier de run.

