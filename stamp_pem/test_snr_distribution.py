import core as pem
import numpy as np
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import scipy.special

data = TimeSeries(np.random.normal(0, 1, 16384*6), sample_rate=16384)
data2 = TimeSeries(np.random.normal(0, 1, 16384*6), sample_rate=16384)
snr = pem.stamp_snr(data, data2, 1)

N = 2.
sigma1 = 1.
sigma2 = 1.
constant = (N**(2. * N) / (2.**((2. * N) - 1) * sigma2**((4. * N) + 2.)))

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
print 'pdf sums to : ' + str(tot)
print 'int_vals sums to : ' + str(np.sum(int_vals))
bins_new = (bins[1:] + bins[:-1]) / 2
cut = (z.size - bins_new.size) / 2

fig = plt.figure()
plt.plot(bins_new, pdf, c='r', label='pixel distribution')
plt.plot(z, int_vals, c='b', label='theory')
plt.plot(z[cut:-cut], np.abs(int_vals[cut:-cut] - pdf),
         c='g', label='residual')
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=12)
ax.set_xlim(-5, 5)
ax.set_yscale('log')
ax.set_ylim(1e-6, .1)
plt.show()
plt.close()
