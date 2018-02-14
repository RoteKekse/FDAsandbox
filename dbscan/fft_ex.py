print(__doc__)
from scipy.fftpack import fft
import numpy as np
import random

N = 672
# sample spacing
T = 1.0/96
x = np.linspace(0.0, N*T, N)
noise = 0.04*np.asarray(random.sample(range(0,1000),672))
y = 50*np.sin(2.0*np.pi*x) + 100 + noise
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)


fraud = np.zeros(672)
pos = np.arange(0,168)*4
fraud[pos] = 40
y_fraud = 50*np.sin(2.0*np.pi*x) + 100 + noise + fraud
yf_fraud = fft(y_fraud)


import matplotlib.pyplot as plt

plt.subplot(4, 1, 1)
plt.title('Oszilierende Daten, z.B. Internetverkehr')
plt.xlabel('Zeit in Tagen')
plt.xlabel('Kumulierte Anzahl von Aktionen')
plt.plot(x,y,'ro')
plt.grid()


plt.subplot(4, 1, 2)
plt.title('Oszilationen in den Daten')
plt.xlabel('Frequenzen')
plt.xlabel('Frequenzstärke')
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),'ro')
plt.grid()

plt.subplot(4, 1, 3)
plt.title('Oszilierende Daten manipuliert durch stündliche Erhöhung der Aktionen, z.B. Internetverkehr')
plt.xlabel('Zeit in Tagen')
plt.xlabel('Kumulierte Anzahl von Aktionen')
plt.plot(x,y_fraud,'ro')
plt.grid()


plt.subplot(4, 1, 4)
plt.title('Oszilationen in den manipulierten Daten')
plt.xlabel('Frequenzen')
plt.xlabel('Frequenzstärke')
plt.plot(xf, 2.0/N * np.abs(yf_fraud[0:N//2]),'ro')
plt.ylim(0,60)
plt.grid()
plt.show()

