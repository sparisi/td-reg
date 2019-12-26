import os, errno, time, sys

name = sys.argv[1] # name of the script to run

for i in range(1,51):
    for j in [0, 1]:
        for k in [0, 1]:
            text = """\
#!/bin/bash

# job name
#SBATCH -J job_name

# logfiles
#SBATCH -o log/stdout_""" + name + """_""" + str(i) + """_""" + str(j) + """_""" + str(k) + """\
#SBATCH -e log/stderr_""" + name + """_""" + str(i) + """_""" + str(j) + """_""" + str(k) + """\

# request computation time hh:mm:ss
#SBATCH -t 8:00:00

# request virtual memory in MB per core
#SBATCH --mem-per-cpu=1000

# nodes for a single job
#SBATCH -n 1

#SBATCH -C avx2
#SBATCH -c 4

module load matlab
matlab -nosplash -nodesktop -nodisplay -r "INSTALL; cd pendulum; """ + name + """(""" + str(i) + """,""" + str(j) + """,""" + str(k) + """); exit"
    """

            text_file = open('r.sh', "w")
            text_file.write(text)
            text_file.close()

            os.system('sbatch r.sh')
