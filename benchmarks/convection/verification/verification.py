import os
import matplotlib.pyplot as plt
import numpy as np
import re
import shutil
import subprocess

to_parse = ['JZ_v100', 'JZ_a100', 'JZ_csl', 'AA_genoa_sockets', 'AA_genoa_CCDs', 'AA_mi250']
to_really_parse = []

verification_py_dir = os.path.abspath(os.path.dirname(__file__))

for p in to_parse:
    try:
        os.listdir(p)
        to_really_parse.append(p)
    except:
        pass

print('Present directories : ' + ' '.join(to_really_parse))

for p in to_really_parse:
    print(f'Analysing {p}')

    # Generate logs if necessary
    path = os.path.join(p, 'logs')
    if os.path.exists(path):
        shutil.rmtree(os.path.join(p, 'logs'))
    os.chdir(p)
    os.mkdir('logs')
    try:
        print(os.getcwd())
        subprocess.run([f'{verification_py_dir}/dump_logs.sh'])
    except:
        print('Something wrong happened when trying to read the logs ! Investigate further !')
        exit(1)
    os.chdir('logs')

    run_list = {'invalid':[]}
    for f in os.listdir('.'):
        if f.startswith('run') and not 'verif' in f:
            try:
                matching = re.match('run_N([0-9]+)x([0-9])+', f)
                N_nodes = int(matching.group(1))
                process_per_node = int(matching.group(2))
            except:
                print(f'Error while trying to analyze {f}. Skipping')
                continue
            
            N_proc = N_nodes * process_per_node
            rname = f'{p}/{N_nodes}x{process_per_node}'

            # Looking for bad values, and weird dts
            f_in = open(f, 'r')
            time = []
            dt  = []
            mass_max_diff = 1e-999
            for line in f_in:
                if 'Negative density detected' in line or 'Negative pressure detected' in line:
                    run_list['invalid'].append((rname, 'Negative density or pressure'))
                    break

                if 'terminate called after throwing an instance' in line:
                    run_list['invalid'].append((rname, 'Exception thrown during execution'))
                    break

                if 'scalar_data' in line and not 'Output' in line:
                    matching = re.match('.*scalar_data : iter=([0-9]+) dt=(.+) time=(.+)', line)
                    try:
                        time.append(float(matching.group(3)))
                        dt.append(float(matching.group(2)))
                    except:
                        print(f' . Problem with line {line} in file {f}')
                        exit(1)

                    # TODO -> Invalidate run if dt is too small wrt to criterion
                if 'Mass' in line:
                    matching = re.match('.*Mass  , (.+) : (.+), (.+), (.+), (.+)', line)
                    try:
                        iter = float(matching.group(1))
                        m = float(matching.group(2))
                        dm_abs = float(matching.group(3))
                        dm_rel = float(matching.group(4))
                        dm_total_abs = float(matching.group(5))
                    except:
                        print(f' . Problem with line {line} in file {f}')
                        exit(1)

                    mass_max_diff = max( mass_max_diff, dm_rel )

                    epsilon_mass = 1e-10
                    if( abs(dm_rel) > epsilon_mass):
                        run_list['invalid'].append((rname, f'Mass not conserved : rel diff {dm_rel} > {epsilon_mass} at iteration {iter}'))
                        break

                    
            f_in.close()

            max_diff_print = mass_max_diff if (mass_max_diff!=1e-999) else "err"
            # Valid run we store the times and dts
            if rname not in (x[0] for x in run_list['invalid']):
                run_list[rname] = (time, dt)
                print( f"Run {rname} is valid : mass deviation = {max_diff_print}" )
            else:
                print( f"Error during Run {rname} mass deviation = {max_diff_print}" )
    
    # We have finished parsing the files, now we display the results
    N_valid = len( run_list ) - 1
    N_invalid = len(run_list['invalid'])
    print(f' . Number of invalid runs : {N_invalid}/{N_invalid+N_valid}')
    for inv_run in run_list['invalid']:
        run, reason = inv_run
        print(f'   -> {run} is invalid for reason {reason}')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sorted_rlist = sorted(run_list.keys())
    for run in sorted_rlist:
        if run == 'invalid':
            continue
        time, dt = run_list[run]
        plt.plot(time, dt, label=f'{run}')
        
    os.chdir('../..')
    plt.xlabel('Time')
    plt.ylabel('dt')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'verif_{p}.png')



            


