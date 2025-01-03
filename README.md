# Running Instructions

NOTE: this is made for a specific dataset we're using, I'll make it more generalized in the future

You need 4 tasks and 4 GPUS. I hard coded it to be 4 tasks, if you want 
you can change it but I don't recommend it. 4 is good.

what i did
```
srun --ntasks=4 --partition=gpu --gres=gpu:4 --mem=100G --pty bash
```

100G is to be safe. You can try it with less.

```
module load mpi/openmpi-4.1.6
module load python/3.11
```

Do you have a virtual environment? you might want one

```
pip install -r requirements.txt
```

to run parallel:

```
mpiexec -n 4 parallel.py
```

to run serial

```
python serial.py
```

be patient, good things take time. Let me know if you have any issues!
