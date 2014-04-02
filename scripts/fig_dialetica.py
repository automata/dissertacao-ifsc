# encoding: utf-8

import numpy as np
import pylab as py
from scipy.optimize import fsolve

# temos p_i, p_j e p_k (tese, antitese e sintese)
p_i = (0.5, 2.0)
p_j = (5.0, 1.0)
p_k = (4.5, 2.5)

# calculamos então a reta que passa pelos pontos p_i e p_j
# para isso, encontramos o coeficiente angular dessa reta
# através de: m = (yb - ya) / (xb - xa)
a = p_i
b = p_j
mr = (b[1] - a[1]) / (b[0] - a[0])
# escolhemos um dos pontos a ou b e aplicamos novamente: y - ya = m(x - xa)
def f(x, ponto, m):
    return ponto[1] + m*(x - ponto[0])

xs = [p_i[0], p_j[0]]
py.plot(xs, [f(x, a, mr) for x in xs], 'k-')

# agora vamos calcular a bissetriz que passa pelo ponto médio entre p_i e p_j
# então, antes de tudo, calculamos o ponto médio pm:
pm = ((a[0] + b[0])/2, (a[1] + b[1])/2)
py.plot(pm[0], pm[1], 'ow')

# e precisamos que ela seja perpendicular, então essa relação precisa ser respeitada:
ms = -1./mr
xs = np.arange(p_i[0], p_j[0])
py.plot(xs, [f(x, pm, ms) for x in xs], 'k:')

# agora a distância de p_k à bissetriz
# então precisamos da reta que passe pelo ponto p_k e seja perpendicular à bissetriz
mt = -1./ms

# e precisamos do ponto de intersecção entre esta reta e a bissetriz
intersec_x = fsolve(lambda x: f(x, pm, ms) - f(x, p_k, mt), 0.0)[0]
intersec_y = f(intersec_x, pm, ms)
py.plot(intersec_x, intersec_y, 'ow')

# plotamos a distância (esta reta t)
xs = [intersec_x, p_k[0]]
py.plot(xs, [f(x, p_k, mt) for x in xs], 'k--')

# plotamos o vetor de p_j à p_k (o vetor)
xs = [p_j[0], p_j[1]]
py.plot([p_j[0], p_k[0]], [p_j[1], p_k[1]], 'k-')

# e plotamos os pontos p_i, p_j, p_k
py.plot(p_i[0], p_i[1], 'ok')
py.text(p_i[0]-.25, p_i[1]-.25, r'$\vec{p_i}$', fontsize=18)
py.plot(p_j[0], p_j[1], 'ok')
py.text(p_j[0]+.2, p_j[1]-.2, r'$\vec{p_j}$', fontsize=18)
py.plot(p_k[0], p_k[1], 'ok')
py.text(p_k[0]+.2, p_k[1]+.2, r'$\vec{p_k}$', fontsize=18)
py.text((intersec_x+p_k[0])/2, (intersec_y+p_k[1])/2+.2, r'$d_{i\to k}$', fontsize=18)
py.text(2.8, 4., r'$B_{i,j}$', fontsize=18)

# e as flechas que simbolizam os vetores
py.arrow(p_i[0], p_i[1], p_j[0]-p_i[0], p_j[1]-p_i[1], head_width=0.15, head_length=0.15, fc='k', length_includes_head=True, overhang=.1)
py.arrow(p_j[0], p_j[1], p_k[0]-p_j[0], p_k[1]-p_j[1], head_width=0.15, head_length=0.15, fc='k', length_includes_head=True, overhang=.1)


py.xlim((0,6.))
py.ylim((0,5.))
py.show()
