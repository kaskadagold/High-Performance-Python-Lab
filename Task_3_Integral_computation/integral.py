
import numpy as np
from mpi4py import MPI
import time

def f(x):
    return np.sin(x)

def trapezoidal(x):
    a = x[0]
    b = x[-1]
    N = len(x)
    value = 0
    for i in range(1, N):
        value += (f(x[i]) + f(x[i - 1])) * (x[i] - x[i - 1]) / 2
    return value
    
time_start = time.time()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

a = 0
b = np.pi
N = 10**5 #number of points
x = np.linspace(a, b, N)

index = int(N / size)
index_rank = x[(rank * index):(index * (rank + 1))]

integral_part = trapezoidal(index_rank)

integral = comm.gather(integral_part, root = 0)

if rank == 0:
    print(np.sum(integral))
    work_time = time.time() - time_start
    print(work_time)
