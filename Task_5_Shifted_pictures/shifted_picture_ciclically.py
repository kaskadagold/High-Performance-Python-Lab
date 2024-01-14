
from PIL import Image
import numpy as np
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-inp", "--input", type = str, default = 'no_name_input', help = "Name of input file.")
parser.add_argument("-o", "--output", type = str, default = 'no_name_output', help = "Name of output file.")
parser.add_argument("-s", "--shift", type = int, default = 1, help = "How many pixels should be shifted.")
args = parser.parse_args()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

image_name = args.input
print(image_name)
image = Image.open(image_name)
arr_image = np.asarray(image)

height = arr_image.shape[0]
width = arr_image.shape[1]

index = np.linspace(0, width, size + 1, dtype = int)
width_rank = index[rank + 1] - index[rank]
image_rank = arr_image[:, index[rank]:index[rank+1]]
image_temp = np.roll(image_rank, args.shift, axis = 0)

data = comm.gather(image_temp, root = 0)
   
if rank == 0:
    value = np.concatenate(tuple(data), axis = 1)   
    pilImage = Image.fromarray(value, mode = "RGBA")
    pilImage.save(args.output)
        
