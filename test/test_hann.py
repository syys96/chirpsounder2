import numpy as n
import scipy.signal as ss
import matplotlib.pyplot as plt

n_samples_per_block = 5e6
wf = n.array(ss.hann(n_samples_per_block),dtype=n.float32)
plt.figure(1)
plt.plot(n.arange(n_samples_per_block), wf)
plt.show()