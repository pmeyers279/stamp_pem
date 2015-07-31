import core as pem
from gwpy.timeseries import TimeSeries
import numpy as np

data = TimeSeries(np.random.normal(0, 1, 16384 * 6), sample_rate=16384)
asd = data.asd(1)
asd_cg = pem.coarseGrain(asd, 64, 64, asd.size / 4)

plot = asd_cg.plot()
plot.show()
