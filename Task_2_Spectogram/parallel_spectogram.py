
import numpy as np
import time
from mpi4py import MPI

def WindowFunction(t, window_position, window_width):
    return np.exp(-(t - window_position)**2 / 2 / window_width**2)

def generation_of_signal(t):
    y = np.sin(t) * np.exp(-t**2 / (2 * 20**2))
    y = y + np.sin(3 * t) * np.exp(-(t - 5 * 2*np.pi)**2 / (2 * 20**2))
    y = y + np.sin(5.5 * t) * np.exp(-(t - 10 * 2*np.pi)**2 / (2 * 5**2))
    return y

def get_specgram(y, t, window_positions, windowsteps, window_width):
    specgram = np.zeros((len(t), windowsteps))

    for i in range(windowsteps):
        sp = np.fft.fft(y * WindowFunction(t, window_positions[i], window_width))
        specgram[:, i] = abs(sp)**2

    return specgram

time_start = time.time()

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

nwindowsteps = 1000
window_width = 2.0 * 2 * np.pi
t = np.linspace(-20 * 2*np.pi, 20 * 2*np.pi, 3**9 + 6)
y = generation_of_signal(t)
window_positions = np.linspace(-20 * 2*np.pi, 20 * 2*np.pi, nwindowsteps)

index = np.linspace(0, nwindowsteps, size + 1, dtype = int)
window_pos_rank = window_positions[index[rank]:index[rank + 1]]

spectogram = get_specgram(y, t, window_pos_rank, len(window_pos_rank), window_width)
    
data = comm.gather(spectogram, root = 0)

if rank == 0:
    work_time = time.time() - time_start
    print(work_time)
