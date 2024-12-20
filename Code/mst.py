import numpy as np
import matplotlib.pyplot as plt

# Zaman dizisi oluşturma
t = np.linspace(0, 2, 1000)

# Giriş sinyallerini oluşturma
giris_ucgen = 2.5 * np.abs(np.mod(t, 1) - 0.5) - 1.25  # Üçgen dalga
giris_kare = 2.5 * (np.mod(t, 1) < 0.5) - 1.25  # Kare dalga

# Integratör devresi çıkışı (üçgen dalganın integrali sinüsoidaldir)
integrator_cikisi = np.cumsum(giris_ucgen) * (t[1] - t[0])

# Türev devresi çıkışı (kare dalganın türevi piktir)
turev_cikisi = np.diff(giris_kare, prepend=giris_kare[0]) / (t[1] - t[0])

# Çizim
plt.figure(figsize=(12, 6))

# Giriş sinyalleri
plt.subplot(2, 2, 1)
plt.plot(t, giris_ucgen, label='Giriş Üçgen Dalga')
plt.title('Giriş Üçgen Dalga')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, giris_kare, label='Giriş Kare Dalga')
plt.title('Giriş Kare Dalga')
plt.legend()

# Çıkış sinyalleri
plt.subplot(2, 2, 3)
plt.plot(t, integrator_cikisi, label='Integratör Çıkışı')
plt.title('Integratör Çıkışı')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, turev_cikisi, label='Türev Çıkışı')
plt.title('Türev Çıkışı')
plt.legend()

plt.tight_layout()
plt.show()
