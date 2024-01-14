
from PIL import Image
import numpy as np
from mpi4py import MPI
import time
import tracemalloc

time_start = time.time()
tracemalloc.start()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

image = Image.open('moscow.png')
arr_image = np.asarray(image)

height = arr_image.shape[0]
width = arr_image.shape[1]

index = np.linspace(0, width, size + 1, dtype = int)
width_rank = index[rank + 1] - index[rank]
image_rank = arr_image[:, index[rank]:index[rank+1]]

image_temp = np.roll(image_rank, 300, axis = 0)

data = comm.gather(image_temp, root = 0)

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak = np.array(comm.gather(peak, root = 0))
   
if rank == 0:
    value = np.concatenate(tuple(data), axis = 1)   
    pilImage = Image.fromarray(value, mode = "RGBA")
    pilImage.save('shifted_picture.png')
    
    work_time = time.time() - time_start
    print(work_time)
    
    print(np.sum(peak / (1024 * 1024)))
