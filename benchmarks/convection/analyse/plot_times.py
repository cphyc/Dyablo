import sys
import csv
import re
import os
import matplotlib.pyplot as plt
import numpy as np

res_dir = sys.argv[1]


timers = [ 'Total/TimeLoop/HydroUpdate_euler ','Total/TimeLoop/Parabolic[explicit] thermal_conduction ','Total/TimeLoop/Parabolic[explicit] viscosity ', 'Total/TimeLoop/AMR ', 'Total/TimeLoop/MPI ghosts ', 'Total/TimeLoop/dt ']
labels = [ "Hydro"                            ,"ThermalConduction"                                     ,'Viscosity'                                    , "AMR"                , 'MPIghosts'                 , "dt"  ]

print("MPI;Nodes;Total(timeloop)", *labels, "Autre", sep=';')

times = {}
run_labels = []
run_MPI = []

for dir in os.listdir(res_dir):
  try:
    matching = re.match('run_N([0-9]+)x([0-9])+', dir)
    N_nodes = int(matching.group(1))
    process_per_node = int(matching.group(2))
  except:
    continue

  filepath = f'{res_dir}/{dir}/timers.txt'
  try:
    f_in = open(filepath, newline='')
    rank_0_times = next(csv.DictReader( f_in, delimiter=';', skipinitialspace=True))
    # Watch out ! final spaces are not skipped in timer names
  except:
    print(f'Error while opening {filepath}. Skipping')
    continue

  timeloop_time = float(rank_0_times["Total/TimeLoop "])
  run_times = [float(rank_0_times[t]) for t in timers]
  other_time = timeloop_time - sum( run_times )
  print( N_nodes*process_per_node, N_nodes, timeloop_time, *run_times, other_time, sep=';' )

  run_labels.append( f'{N_nodes*process_per_node}' )
  run_MPI.append( int(N_nodes*process_per_node) )
  for [label,time] in zip(labels, run_times):
    times.setdefault(label, [])
    times.get(label).append(time)
  times.setdefault("Autre", [])
  times.get("Autre").append(other_time)

reorder = np.argsort( run_MPI )
for timer in times.keys():
  times[timer] = [ times[timer][x] for x in reorder ]
run_labels = [ run_labels[x] for x in reorder ]
run_MPI = [ run_MPI[x] for x in reorder ]


fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.yaxis.grid(True)
ax.xaxis.grid(False)

cumulated_time = np.zeros( len(run_labels) )
for label, times in times.items():
  bars = ax.bar(run_labels, times, bottom=cumulated_time, label=label)
  cumulated_time = cumulated_time + times

for [bar,time] in zip(bars,cumulated_time):
  ax.text(
      bar.get_x() + bar.get_width() / 2,
      time + 0.3,
      round(time, 1),
      horizontalalignment='center',
      weight='bold'
  )

plt.title( f"Dyablo Weak Scaling {res_dir}" )
plt.xlabel('MPI processes')
plt.ylabel('runtime (s)')
plt.legend(title="Timers", reverse=True)
plt.tight_layout()

plt.savefig(f'weak_scaling_times_{res_dir}.png')

plt.show()




  

