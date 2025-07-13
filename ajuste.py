import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, kstest
import random

filename = 'longitudes.txt'

with open(filename, 'r') as f:
    data = [float(line.strip()) for line in f if line.strip()]



# tomo submuestra

submuestra = random.sample(list(data), 500)
# Ajuste gaussiano centrado en la media
mu, std = norm.fit(data)
xmin, xmax = min(data), max(data)
x = np.linspace(xmin, xmax, 100)

# Histograma no normalizado
counts, bins, _ = plt.hist(data, bins=20, density=False, alpha=0.6, color='g', label='Histograma de frecuencas')

# Curva gaussiana centrada en la media
bin_width = bins[1] - bins[0]
p = norm.pdf(x, mu, std) * len(data) * bin_width
plt.plot(x, p, 'r-', label='Ajuste gaussiano')

plt.xlabel('Longitud [$^\circ$]')
plt.ylabel('Frecuencia')
plt.title('Histograma y ajuste de longitudes')
plt.legend()
plt.show()

# Test de normalidad
stat, p_value = shapiro(submuestra)
resultado = f"Shapiro-Wilk p-value: {p_value}\n"
if p_value > 0.05:
    resultado += "No se puede rechazar la hipótesis de normalidad (parece gaussiana).\n"
else:
    resultado += "Se rechaza la hipótesis de normalidad (no parece gaussiana).\n"

# Test de Kolmogorov-Smirnov
d_stat, ks_p_value = kstest(submuestra, 'norm', args=(mu, std))
resultado += f"Kolmogorov-Smirnov p-value: {ks_p_value}\n"
if ks_p_value > 0.05:
    resultado += "No se puede rechazar la hipótesis de normalidad (KS).\n"
else:
    resultado += "Se rechaza la hipótesis de normalidad (KS).\n"


with open("resultado_normalidad.txt", "w") as f:
    f.write(resultado)
    f.write(f"Valor mínimo: {min(list(data))}\n")
    f.write(f"Valor máximo: {max(list(data))}\n")
    f.write(f"\n Media : {mu}\n")
    f.write(f"Desviación estándar (): {std}\n")
