# encoding: utf-8

import numpy as np
import pylab as py
from scipy.optimize import fsolve

# temos p_i, p_j (tese, antitese)
p_i = (1.0, 2.0)
r_i = (5.0, 3.5)
p_j = (5.0, 1.0)

# calculamos então a reta que passa pelos pontos p_i e r_i
# para isso, encontramos o coeficiente angular dessa reta
# através de: m = (yb - ya) / (xb - xa)
a = p_i
b = r_i
mr = (b[1] - a[1]) / (b[0] - a[0])
# escolhemos um dos pontos a ou b e aplicamos novamente: y - ya = m(x - xa)
def f(x, ponto, m):
    return ponto[1] + m*(x - ponto[0])

xs = np.arange(0., 7.)
py.plot(xs, [f(x, a, mr) for x in xs], 'k:')

# plotamos o ponto medio a_i
py.plot((p_i[0]+r_i[0])/2, (p_i[1]+r_i[1])/2, 'ok')
py.text((p_i[0]+r_i[0])/2, (p_i[1]+r_i[1])/2-.5, r'$\vec{a_j}$', fontsize=18)

# agora a distância de p_j à reta que passa entre p_i e r_i, perpendicular
ms = -1./mr

# e precisamos do ponto de intersecção entre esta reta e a bissetriz
intersec_x = fsolve(lambda x: f(x, p_i, mr) - f(x, p_j, ms), 0.0)[0]
intersec_y = f(intersec_x, p_i, mr)
py.plot(intersec_x, intersec_y, 'ow')

# plotamos a distância s_{i,j}
xs = [intersec_x, p_j[0]]
py.plot(xs, [f(x, p_j, ms) for x in xs], 'k--')
# # plotamos o vetor de p_j à p_k (o vetor)
# xs = [p_j[0], p_j[1]]
# py.plot([p_j[0], p_k[0]], [p_j[1], p_k[1]], 'k-')

# e plotamos os pontos p_i, p_j, r_i
py.plot(p_i[0], p_i[1], 'ok')
py.text(p_i[0]-.25, p_i[1]-.5, r'$\vec{p_i}$', fontsize=18)
py.plot(p_j[0], p_j[1], 'ok')
py.text(p_j[0]+.2, p_j[1]-.2, r'$\vec{p_j}$', fontsize=18)
py.plot(r_i[0], r_i[1], 'ok')
py.text(r_i[0], r_i[1]-.5, r'$\vec{r_i}$', fontsize=18)

py.text(5.5, 4., r'$L_i$', fontsize=18)
py.text((intersec_x+p_j[0])/2, (intersec_y+p_j[1])/2+.1, r'$s_{i,j}$', fontsize=18)
py.text(3., 3.2, r'$\vec{D_j}$', fontsize=18)


# e as flechas que simbolizam os vetores
py.arrow(p_i[0], p_i[1], p_j[0]-p_i[0], p_j[1]-p_i[1], head_width=0.15, head_length=0.15, fc='k', length_includes_head=True, overhang=.1)
py.arrow(p_i[0], p_i[1]+.25, r_i[0]-p_i[0], r_i[1]-p_i[1], head_width=0.15, head_length=0.15, fc='k', length_includes_head=True, overhang=.1)


py.xlim((0,6.))
py.ylim((0,5.))
py.xlabel('x')
py.ylabel('y')
py.show()
