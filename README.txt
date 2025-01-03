let's do this one last time

You need 4 tasks and 4 GPUS lol. I hard coded it to be 4 tasks, if you want 
you can change it but I don't recommend it. 4 is good.

what i did
srun --ntasks=4 --partition=gpu --gres=gpu:4 --mem=100G --pty bash

100G is to be safe. You can try it with less. It was available when I ran so 
I went with it. Does that make me a bad person? maybe.... but sometimes you
have to be a little evil to do some good

thennn

module load mpi/openmpi-4.1.6
module load python/3.11

Do you have a virtual environment? you might want one

pip install -r requirements.txt

Now we're cooking

to run parallel:
mpiexec -n 4 parallel.py

to run serial
python serial.py

be patient, good things take time. Let me know if you have any issues! it defintiely works! :c
