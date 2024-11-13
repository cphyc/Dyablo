#!/bin/python

import shutil
from math import log2
import os
import sys
import subprocess

available_targets = ('JZ_v100', 'JZ_a100', 'JZ_csl', 'AA_genoa_sockets','AA_genoa_CCDs', 'AA_mi250')
def usage():
    print(f'Usage : python3 {sys.argv[0]} TARGET [--run]')
    print('With target in the following : ' + ' '.join(available_targets))
    exit(1)

execute_sbatch = "--run" in sys.argv
    
if len(sys.argv) < 2:
    usage()
    
target = sys.argv[1]
if target not in available_targets:
    usage()

job_tmpl=f'job_tmpl_{target}.slurm'

if target == 'JZ_v100':
    machine_threads_per_node=40
    machine_gpus_per_nodes=4
    executable_path='./dyablo_v100'
    account='jza'
elif target == 'JZ_a100':
    machine_threads_per_node=64
    machine_gpus_per_nodes=8
    executable_path='./dyablo_a100'
    account='jza'
elif target == 'JZ_csl':
    machine_threads_per_node=80 # 80 with multithreading; 40 otherwise
    machine_gpus_per_nodes=0
    executable_path='./dyablo_csl'
    account='zah'
elif target == 'AA_genoa_sockets':
    machine_threads_per_node=384
    machine_gpus_per_nodes=0
    executable_path='./dyablo_genoa'
    account='cin4545'
elif target == 'AA_genoa_CCDs':
    machine_threads_per_node=384
    machine_gpus_per_nodes=0
    executable_path='./dyablo_genoa'
    account='cin4545'
elif target == 'AA_mi250':
    machine_threads_per_node=128
    machine_gpus_per_nodes=8
    executable_path='./dyablo_mi250'
    account='aim1623'

template_dir=os.path.dirname(os.path.realpath(__file__))

# Replace keys from params in src with values from params and write to dst
def replace_in_file(src_filename, dst_filename, params):
    f_in  = open( src_filename, "rt" )
    data = f_in.read()
    f_in.close()

    for key in params.keys() :
      data = data.replace( key, str(params[key]) )

    os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
    f_out = open( dst_filename, "wt"  )
    f_out.write(data)
    f_out.close()

def safe_symlink(filename, dst_dir):
  # Safely create a symlink to local file
  try:
    os.unlink(os.path.join(dst_dir, filename))
  except FileNotFoundError:
    pass  
  if not os.path.isfile( os.path.realpath(filename) ):
     raise FileNotFoundError(f"Could not find file {filename}")     
  os.symlink(os.path.realpath(filename), os.path.join(dst_dir, filename))

def run_testcase( nb_nodes, mpi_per_node, nrep_x, nrep_y ):
  if( mpi_per_node*nb_nodes != nrep_x*nrep_y ):
    raise ValueError("Number of replications doesn't match number of process")


  dst_dir = f"{target}/run_N{nb_nodes}x{mpi_per_node}_R{nrep_x}x{nrep_y}"
  # Create restart.ini
  ini_src = os.path.join(template_dir, "restart_tmpl.ini") 
  ini_dst = os.path.join(dst_dir, "restart.ini")

  lmin=3 + int(log2(max(nrep_x, nrep_y)))
  lmax=lmin+4
  cor_x = nrep_x*8
  cor_y = nrep_y*8

  ini_params = {}
  ini_params["<nrep_x>"] = nrep_x
  ini_params["<nrep_y>"] = nrep_y
  ini_params["<xmax>"] = 4.0*nrep_x
  ini_params["<ymax>"] = 4.0*nrep_y
  ini_params["<level_min>"] = lmin
  ini_params["<level_max>"] = lmax
  ini_params["<cor_x>"] = cor_x
  ini_params["<cor_y>"] = cor_y

  replace_in_file(ini_src, ini_dst, ini_params)

  # Create job.slurm
  slurm_src = os.path.join(template_dir, job_tmpl)
  slurm_dst = os.path.join(dst_dir, job_tmpl)

  slurm_params = {}
  slurm_params["<nb_nodes>"] = nb_nodes
  slurm_params["<nb_mpi>"] = nb_nodes*mpi_per_node
  slurm_params["<nb_gpus>"] = machine_gpus_per_nodes
  slurm_params["<nb_threads>"] = int(machine_threads_per_node/mpi_per_node)
  slurm_params["<account>"] = account

  replace_in_file(slurm_src, slurm_dst, slurm_params)

  # Make symlinks to executable and restart.ini
  safe_symlink(executable_path, dst_dir)
  safe_symlink('restart.h5', dst_dir)

  if execute_sbatch :
      cmd = ["sbatch", job_tmpl]
      p = subprocess.Popen(cmd, cwd=dst_dir)
      p.wait()

if target == 'JZ_v100':
    run_testcase( nb_nodes=1, mpi_per_node=1, nrep_x=1, nrep_y=1 )
    run_testcase( nb_nodes=1, mpi_per_node=2, nrep_x=2, nrep_y=1 )
    run_testcase( nb_nodes=1, mpi_per_node=4, nrep_x=2, nrep_y=2 )
    run_testcase( nb_nodes=2, mpi_per_node=4, nrep_x=4, nrep_y=2 )
    run_testcase( nb_nodes=4, mpi_per_node=4, nrep_x=4, nrep_y=4 )
    run_testcase( nb_nodes=8, mpi_per_node=4, nrep_x=8, nrep_y=4 )
    run_testcase( nb_nodes=16, mpi_per_node=4, nrep_x=8, nrep_y=8 )
    run_testcase( nb_nodes=32, mpi_per_node=4, nrep_x=16, nrep_y=8 )
    run_testcase( nb_nodes=64, mpi_per_node=4, nrep_x=16, nrep_y=16 )
elif target == 'JZ_a100':
    run_testcase( nb_nodes=1, mpi_per_node=1, nrep_x=1, nrep_y=1 )
    run_testcase( nb_nodes=1, mpi_per_node=2, nrep_x=2, nrep_y=1 )
    run_testcase( nb_nodes=1, mpi_per_node=4, nrep_x=2, nrep_y=2 )
    run_testcase( nb_nodes=1, mpi_per_node=8, nrep_x=4, nrep_y=2 )
    run_testcase( nb_nodes=2, mpi_per_node=8, nrep_x=4, nrep_y=4 )
    run_testcase( nb_nodes=4, mpi_per_node=8, nrep_x=8, nrep_y=4 )
    run_testcase( nb_nodes=8, mpi_per_node=8, nrep_x=8, nrep_y=8 )
    run_testcase( nb_nodes=16, mpi_per_node=8, nrep_x=16, nrep_y=8 )
    run_testcase( nb_nodes=32, mpi_per_node=8, nrep_x=16, nrep_y=16 )
elif target == 'JZ_csl':
    run_testcase( nb_nodes=1, mpi_per_node=2, nrep_x=2, nrep_y=1 )
    run_testcase( nb_nodes=2, mpi_per_node=2, nrep_x=2, nrep_y=2 )
    run_testcase( nb_nodes=4, mpi_per_node=2, nrep_x=4, nrep_y=2 )
    run_testcase( nb_nodes=8, mpi_per_node=2, nrep_x=4, nrep_y=4 )
    run_testcase( nb_nodes=16, mpi_per_node=2, nrep_x=8, nrep_y=4 )
    run_testcase( nb_nodes=32, mpi_per_node=2, nrep_x=8, nrep_y=8 )
    run_testcase( nb_nodes=64, mpi_per_node=2, nrep_x=16, nrep_y=8 )
    run_testcase( nb_nodes=128, mpi_per_node=2, nrep_x=16, nrep_y=16 )
    run_testcase( nb_nodes=256, mpi_per_node=2, nrep_x=32, nrep_y=16 )

    #run_testcase( nb_nodes=1, mpi_per_node=1, nrep_x=1, nrep_y=1 )
    #run_testcase( nb_nodes=2, mpi_per_node=1, nrep_x=2, nrep_y=1)
    #run_testcase( nb_nodes=4, mpi_per_node=1, nrep_x=2, nrep_y=2)
    #run_testcase( nb_nodes=8, mpi_per_node=1, nrep_x=4, nrep_y=2)
    #run_testcase( nb_nodes=16, mpi_per_node=1, nrep_x=4, nrep_y=4)
    #run_testcase( nb_nodes=32, mpi_per_node=1, nrep_x=8, nrep_y=4)
    #run_testcase( nb_nodes=64, mpi_per_node=1, nrep_x=8, nrep_y=8)
    #run_testcase( nb_nodes=128, mpi_per_node=1, nrep_x=16, nrep_y=8)
    #run_testcase( nb_nodes=256, mpi_per_node=1, nrep_x=16, nrep_y=16)
    #run_testcase( nb_nodes=512, mpi_per_node=1, nrep_x=32, nrep_y=16)
elif target == 'AA_genoa_sockets':                                                                               
    # 1 Process per CPU                                                                                  
    run_testcase( nb_nodes=1, mpi_per_node=2, nrep_x=2, nrep_y=1 )                                      
    run_testcase( nb_nodes=2, mpi_per_node=2, nrep_x=2, nrep_y=2 )                                      
    run_testcase( nb_nodes=4, mpi_per_node=2, nrep_x=4, nrep_y=2 )                                      
    run_testcase( nb_nodes=8, mpi_per_node=2, nrep_x=4, nrep_y=4 )                                      
    run_testcase( nb_nodes=16, mpi_per_node=2, nrep_x=8, nrep_y=4 )                                     
    run_testcase( nb_nodes=32, mpi_per_node=2, nrep_x=8, nrep_y=8 )                                     
    run_testcase( nb_nodes=64, mpi_per_node=2, nrep_x=16, nrep_y=8 )                                    
    run_testcase( nb_nodes=128, mpi_per_node=2, nrep_x=16, nrep_y=16 )                                  
    run_testcase( nb_nodes=256, mpi_per_node=2, nrep_x=32, nrep_y=16 )
    run_testcase( nb_nodes=512, mpi_per_node=2, nrep_x=32, nrep_y=32 )
elif target == 'AA_genoa_CCDs':                                                                               
    # 4 Process per CPU                                                                                  
    run_testcase( nb_nodes=1, mpi_per_node=8, nrep_x=4, nrep_y=2 )                                      
    run_testcase( nb_nodes=2, mpi_per_node=8, nrep_x=4, nrep_y=4 )                                      
    run_testcase( nb_nodes=4, mpi_per_node=8, nrep_x=8, nrep_y=4 )                                      
    run_testcase( nb_nodes=8, mpi_per_node=8, nrep_x=8, nrep_y=8 )                                      
    run_testcase( nb_nodes=16, mpi_per_node=8, nrep_x=16, nrep_y=8 )                                     
    run_testcase( nb_nodes=32, mpi_per_node=8, nrep_x=16, nrep_y=16 )                                     
    run_testcase( nb_nodes=64, mpi_per_node=8, nrep_x=32, nrep_y=16 )                                    
    run_testcase( nb_nodes=128, mpi_per_node=8, nrep_x=32, nrep_y=32 )                                  
    run_testcase( nb_nodes=256, mpi_per_node=8, nrep_x=64, nrep_y=32 )
    run_testcase( nb_nodes=512, mpi_per_node=8, nrep_x=64, nrep_y=64 )
elif target == 'AA_mi250':                                                                               
    run_testcase( nb_nodes=1, mpi_per_node=1, nrep_x=1, nrep_y=1 )                                       
    run_testcase( nb_nodes=1, mpi_per_node=2, nrep_x=2, nrep_y=1 )                                       
    run_testcase( nb_nodes=1, mpi_per_node=4, nrep_x=2, nrep_y=2 )                                       
    run_testcase( nb_nodes=1, mpi_per_node=8, nrep_x=4, nrep_y=2 )                                       
    run_testcase( nb_nodes=2, mpi_per_node=8, nrep_x=4, nrep_y=4 )                                       
    run_testcase( nb_nodes=4, mpi_per_node=8, nrep_x=8, nrep_y=4 )                                       
    run_testcase( nb_nodes=8, mpi_per_node=8, nrep_x=8, nrep_y=8 )                                       
    run_testcase( nb_nodes=16, mpi_per_node=8, nrep_x=16, nrep_y=8 )                                     
    run_testcase( nb_nodes=32, mpi_per_node=8, nrep_x=16, nrep_y=16 )                                    
    run_testcase( nb_nodes=64, mpi_per_node=8, nrep_x=32, nrep_y=16 )                                    
    run_testcase( nb_nodes=128, mpi_per_node=8, nrep_x=32, nrep_y=32 )                                   
    run_testcase( nb_nodes=256, mpi_per_node=8, nrep_x=64, nrep_y=32 )
    
