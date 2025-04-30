# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:09:02 2025

@author: Namicano
"""

import matplotlib.pyplot as plt

# Dados de exemplo (substitua pelos seus dados reais)
tempos = [0, 1, 2, 3, 4, 5]  # Unidades de tempo (quadros de vídeo ou áreas)
afideos_detectados = [10,150, 25, 80, 120, 50]  # Número de afídeos detectados
perda_rendimento = [0.01, 0.05, 0.15, 0.30, 0.50, 0.70] # Perda de rendimento estimada

# Níveis de dano (substitua pelos seus valores reais)
NE = 20  # Nível de Equilíbrio
NC = 60  # Nível de Controle
NDE = 100 # Nível de Dano Econômico

# Criar o gráfico
plt.plot(tempos, afideos_detectados, label='Afídeos Detectados')
plt.axhline(y=NE, color='green', linestyle='--', label='Nível de Equilíbrio')
plt.axhline(y=NC, color='orange', linestyle='--', label='Nível de Controle')
plt.axhline(y=NDE, color='red', linestyle='--', label='Nível de Dano Econômico')

plt.xlabel('Tempo (ou Área)')
plt.ylabel('Número de Afídeos (ou Perda de Rendimento)')
plt.title('Níveis de Dano Econômico por Afídeos')
plt.legend()
plt.grid(True)
plt.show()