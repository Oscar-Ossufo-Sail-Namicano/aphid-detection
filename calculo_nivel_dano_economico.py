# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:09:02 2025

@author: Namicano
"""

import matplotlib.pyplot as plt

# Dados de exemplo (substitua pelos seus dados reais)
tempos =             [1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]  # Unidades de tempo (quadros de vídeo ou áreas)
afideos_detectados = [10,11, 15, 15, 18, 5, 5, 8, 8, 5, 5, 10, 0,0,0,3,3,9,20,20,20,11,11,15,15,15,14,14,16,17]  # Número de afídeos detectados
perda_rendimento = [0.01, 0.05, 0.15, 0.30, 0.50, 0.70] # Perda de rendimento estimada

# Níveis de dano (substitua pelos seus valores reais)
NE = 10  # Nível de Equilíbrio
NC = 15  # Nível de Controle
NDE = 20 # Nível de Dano Econômico

# Criar o gráfico
plt.plot(tempos, afideos_detectados, label='Afídeos Detectados')
plt.axhline(y=NE, color='green', linestyle='--', label='Nível de Equilíbrio')
plt.axhline(y=NC, color='orange', linestyle='--', label='Nível de Controle')
plt.axhline(y=NDE, color='red', linestyle='--', label='Nível de Dano Econômico')

plt.xlabel('Numero de amostra (folha)')
plt.ylabel('Número de Afídeos (colonias)')
plt.title('Níveis de Dano Econômico por Afídeos')
plt.legend()
plt.grid(True)
plt.show()