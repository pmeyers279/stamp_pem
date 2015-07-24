import core2 as pem
import numpy as np
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import scipy.special

data = TimeSeries(np.random.normal(0, 1, 163840), sample_rate=16384)
data2 = TimeSeries(np.random.normal(0, 1, 163840), sample_rate=16384)
snr = 4*pem.stamp_snr(data, data2, 1)

N = 2.
sigma1 = 1.
sigma2 = 1. 
constant = (N**(2. * N) / (2.**((2. * N) - 1) * sigma2**((4. * N) + 2.)))

print constant

x = np.arange(0.01, 10.01, 0.01)
z = np.arange(-100, 100.01, 0.01)

int_vals = []

for zi in z:
    y = (constant * np.abs(x) * np.exp(-np.abs(x * zi / sigma1**2)) *
         scipy.special.kv(0, N * x / sigma2**2) * x**(2 * N - 1))
    int_vals.append(100 * np.trapz(y, x))
int_vals = np.asarray(int_vals)
int_vals = int_vals / np.sum(int_vals)
pat = np.real(snr.value.reshape(snr.value.size, 1))
n, bins, patches = plt.hist(
    pat, bins=(z[1:] + z[:-1]) / 2)
plt.close()
pdf = n / np.sum(n)

tot = np.sum(pdf)
print 'pdf sums to :' + str(tot)
print 'int_vals sums to :' + str(np.sum(int_vals))

fig = plt.figure()
plt.plot((bins[1:] + bins[:-1]) / 2, pdf, c='r')
plt.plot(z, int_vals, c='b')
ax = plt.gca()
ax.set_xlim(-4, 4)
ax.set_yscale('log')
ax.set_ylim(1e-5, .1)
plt.show()
plt.close()

fig = plt.figure()
plt.plot((bins[1:] + bins[:-1]) / 2, pdf, c='r')
ax = plt.gca()
ax.set_xlim(-2, 2)
plt.show()
plt.close()

fig = plt.figure()
plt.plot(z, int_vals, c='b')
ax = plt.gca()
ax.set_xlim(-2, 2)
plt.show()
plt.close()
