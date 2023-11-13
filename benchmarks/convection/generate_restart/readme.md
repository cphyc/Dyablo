## Préparation des runs de convection

### Run préliminaire - relaxation

Le setup de convection initial est analytique. Il nécessite donc un certain nombre de pas de temps pour converger vers un état stable. Un run préliminaire est donc effectué pour atteindre cet état thermiquement relaxé. Cet état est atteint initialement avec un run en grille fixe à basse résolution (lvl 3) de manière à converger rapidement. L’utilisation d’une grille fixe permet d’accélérer la vitesse de ce calcul, et d’éviter un sur-raffinement au moment du déclenchement de l’instabilité initiale.

Les paramètres du run dans `1_run_prep/run_prep.ini`

### Runs de raffinement

Une fois le premier run de préparation arrivé à l’équilibre, on va raffiner progressivement le domaine en effectuant des restarts successifs et en augmentant de 2 niveaux à chaque fois sur une centaine d’itérations. Le raffinement est effectué à chaque itération et le load-balancing est effectué toutes les 50 itérations (soit deux fois dans le run). Cela permet à Dyablo de raffiner le domaine et de revenir à l’équilibre thermique. On applique cette procédure deux fois pour obtenir un fichier de restart allant de `level_min=3` à `level_max=7`. 

* `2_refine_1/refine_1.ini`
* `3_refine_2/refine_2.ini`

Note : ces fichiers .ini supposent que le résultat de l'étape précédente (`restart_C91_xxxxxx.h5`) est copiée sous le nom `restart.h5` dans le dossier 2_refine_1 puis 3_refine_2 


### Run préliminaire de référence

**Le résultat du run préliminaire (`3_refine_2/restart_C91_100.h5`) peut (et doit) être réutilisé entre les runs et entre les architectures**

Un run de référence a été exécuté sur la partition V100 de Jean Zay : job_init_jz.slurm

Pour executer le run préliminaire, il faut créer un lien `dyablo_v100` vers l'executable dyablo dans le dossier generate_restart. 

Adaptez le script job_init_jz.slurm à la machine et à vos allocations d'heures de calcul


