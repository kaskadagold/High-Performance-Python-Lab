
import numpy as np
from mpi4py import MPI
import time

def formula(r, x):
    return r * x * (1 - x)

time_start = time.time()

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

# Parameters of calculation
number_of_steps = 700
last = 200
length = 1000

r = np.linspace(0.8, 4, length)
x0 = np.random.sample() * np.ones(length)
index = np.linspace(0, length, size + 1, dtype = int)

x = []
# Ranging calculations between processes
r_rank = r[index[rank]:index[rank + 1]]
x_rank = x0[index[rank]:index[rank + 1]]

# Calculation in every process
for i in range(number_of_steps):
    x_rank = formula(r_rank, x_rank)
    x.append(x_rank)
    
x_result = x[number_of_steps - last:number_of_steps]

data = comm.gather([x_result, r_rank], root = 0)

if rank == 0:
    work_time = time.time() - time_start
    print(work_time)
