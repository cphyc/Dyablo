import sys
import re

if '--by-time' in sys.argv:
    sort_type = 'by-time'
else:
    sort_type = 'alphabetical'

gdoc_format = '--gdoc-format' in sys.argv
    
f_in = open(sys.argv[1], 'r')
reading = False

timers = []

re_gpu = r'(.+)time .+:(.+)s.+\((.+)\).+,.+:(.+)s.+\((.+)\).*'
re_cpu = r'(.+)time .+:(.+) .+\((.+)\).*'
ret_gpu = '.+:(.+),.+:(.+)'
ret_cpu = '.+:(.+)'
rexp = ''
run_type = ''
for line in f_in:
    if '/bin/bash' in line:
        reading = False

    if reading:
        m = re.match(rexp, line.strip())
        if not m:
            continue
        kernel_name = m.group(1).strip()
        k_cpu_time = m.group(2).strip()
        k_cpu_pct  = m.group(3).strip()

        entry = [kernel_name, k_cpu_time, k_cpu_pct]
        
        if run_type == 'GPU':
            k_gpu_time = m.group(4).strip()
            k_gpu_pct  = m.group(5).strip()
            entry += [k_gpu_time, k_gpu_pct]

        timers.append(entry)

    if 'Total elapsed time' in line:
        reading = True

        m = re.match(ret_gpu, line.strip())
        if not m:
            m = re.match(ret_cpu, line.strip())
            if not m:
                print('Unknown run type ! ERROR')
                exit(1)
            else:
                run_type = 'CPU'
                rexp = re_cpu
        else:
            run_type = 'GPU'
            rexp = re_gpu
        
        print('Run type :', run_type)
        total_read_cpu = float(m.group(1))
        if run_type == 'GPU':
            total_read_gpu = float(m.group(2))

# Alphabetical sort
if gdoc_format:
    timers.sort(key=lambda x:x[0])
    x=timers
    print(f'{total_read_cpu}\t{x[10][1]}\t{x[0][1]}\t{x[5][1]}\t{x[6][1]}\t{x[7][1]}\t{x[8][1]}\t{x[9][1]}\t{x[12][1]}')
else:
    if sort_type == 'alphabetical':
        timers.sort(key=lambda x:x[0])
    else:
        timers.sort(key=lambda x:-float(x[1]))

    if run_type == 'CPU':
        print('Kernel name\tCPU time\tCPU percent'.expandtabs(40))
    else:
        print('Kernel name\tCPU time\tCPU percent\tGPU time\t GPU percent'.expandtabs(40))
    
    for t in timers:
        print(f'\t'.join(t).expandtabs(40))
    
    if run_type == 'GPU':
        print(f'Total measured time on CPU : {total_read_cpu}; on GPU : {total_read_gpu}')
    else:
        print(f'Total measured time on CPU : {total_read_cpu}')

        
        
