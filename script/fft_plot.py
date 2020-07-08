import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import interactive

output_dir = "plot/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

N = 128
x = np.arange(-5, 5, 10./(2 * N))

y = np.exp(-x * x)

y_fft = np.fft.fftshift(np.abs(np.fft.fft(y))) / np.sqrt(len(y))

z = np.zeros(N*2)
z[int(N-(N/4)):int(N+(N/4))] = 1
z_fft = np.fft.fftshift(np.abs(np.fft.fft(z))) / np.sqrt(len(z))

plt.rcParams.update({'font.size': 16})

plt.plot(x,y, 'k')
plt.title("Gaussian function")
plt.savefig(os.path.join(output_dir, 'gaussian.png'), dpi=600)
plt.clf()

plt.plot(x,y_fft, 'k')
plt.xlim(-1, 1)
plt.title("FFT of Gaussian function")
plt.savefig(os.path.join(output_dir, 'gaussian_fft.png'), dpi=600)
plt.clf()

plt.plot(x,z, 'k')
plt.title("Boxcar function")
plt.savefig(os.path.join(output_dir, 'boxcar.png'), dpi=600)
plt.clf()

plt.plot(x,z_fft, 'k')
plt.xlim(-1, 1)
plt.title("FFT of Boxcar function")
plt.savefig(os.path.join(output_dir, 'boxcar_fft.png'), dpi=600)
plt.clf()
