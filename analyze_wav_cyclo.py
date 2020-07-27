import glob

import matplotlib
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import savemat

from flight_acoustics import psd

font = {'family': 'serif',
        'weight': 'normal',
        'size': 10}

plt.style.use('ggplot')
matplotlib.rc('font', **font)
matplotlib.rcParams.update({'mathtext.fontset': 'dejavuserif'})
matplotlib.rcParams.update({'figure.autolayout': True})

xlim = (0, 200)
ylim = (25, 90)

amb, fsamb = sf.read('calibration.wav')
lnom = 70

famb, psdamb, lamb = psd(amb[:, 0], fsamb)

cal = lnom - lamb

famb, psdamb, lamb = psd(amb[:, 0], fsamb, cal)
print('Calibrated level is {:0.1f} with {:0.1f} expected.'.format(lamb, lnom))
print('Calibration factor is {:0.1f} dB.'.format(cal))

files = glob.glob('cyclo_rotor_noise_belt/*.wav')
print(files)

data1, fs = sf.read(files[0])
data2, _ = sf.read(files[1])
data3, _ = sf.read(files[2])

f1, p1, l1 = psd(data1[:, 0], fs, cal)
f2, p2, l2 = psd(data2[:, 0], fs, cal)
f3, p3, l3 = psd(data3[:, 0], fs, cal)

fig, ax = plt.subplots()
ax.plot(f2, p2, label='PA 40, Level: {:0.1f} dB'.format(l2))
ax.plot(f1, p1, label='PA 20, Level: {:0.1f} dB'.format(l1))
ax.plot(f3, p3, label='No Blades, Level: {:0.1f} dB'.format(l3))


ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel('Frequency, Hz')
ax.set_ylabel('SPL, dB')
ax.legend()

fig.savefig('cyclo_rotor_noise_belt/set_2.pdf')
savemat('cyclo_rotor_noise_belt/set_2.mat', {'f1': f1, 'f2': f2, 'f3': f3})
plt.show(block=True)
