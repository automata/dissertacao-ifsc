# encoding: utf-8

# CASO 2: PCA usado apenas para visualização

# 1. normalizo F
# 2. separo as pinturas em 12 classes (Fs)
# 3. ordeno as pinturas em ordem cronológica (Ford)
# 4. calculo os protótipos (ponto médio de cada classe) (Prots)
# 5. calculo PCA dos Prots
# 6. calculo dialética, inovação e oposição dos Prots

import pickle as pk
import numpy as np
import pca as pc
import pylab as py
import config as conf
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as image

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA

# matriz de dados (feature matrix)
f = open('dados/feature_matrix1.pkl', 'rb')
Fa = np.array(pk.load(f))
f.close()

f = open('dados/feature_matrix2.pkl', 'rb')
Fb = np.array(pk.load(f))
f.close()

F = np.concatenate((Fa, Fb), axis=1)

# filtramos a tabela de pinturas, retirando colunas com valores zerados e
# degenerados
cols = [0,1,2] + range(5,12) + range(14,21) + range(23,30) + range(32,51) + range(52,64) + range(65,77) + range(78,90) + range(91,97)+ range(98,103)
F__ = F[:,cols]

# selecionamos as features
#F__ = F__[:,[83,84,85,86,87,88,89]]  # features de curvatura, seg, convex hull
#F__ = F__[:,range(31,43) + range(43,55) + range(55,66) + range(67,78)]  # features de haralick
#F__ = F__[:,[83,84,85,86,87,88,89]]
#F__ = F__[:,[83,84,85,86,87,88,89]+range(31,43) + range(43,55) + range(55,66) + range(67,78)]

# normalizamos
M = np.mean(F__, axis=0)
D = np.std(F__, axis=0)
F = np.nan_to_num((F__-M) / D)
# separamos as pinturas em classes de 20 pinturas, uma para cada pintor
Fs = [F[i:i+20] for i in range(0,240,20)]

# ordenamos a tabela pela ordem correta cronológica dos pintores
Ford = [Fs[conf.ordem[i]] for i in range(12)]

# calculamos os protótipos (pontos médios da nuvem de pinturas)
Prots = []
for j in xrange(len(Ford)):
    # protótipo (ponto médio), média de cada dimensão
    prot = np.array([np.mean(Ford[j][:,k]) for k in range(Ford[j].shape[1])])
    prot = np.nan_to_num(prot)
    Prots.append(prot)
# ordenamos os prots
Prots = [Prots[conf.ordem[i]] for i in range(12)]

# dados usados para cálculo dos 'metrics'
agents = [conf.artistas[conf.ordem[i]] for i in range(12)]
dados = np.array(Prots)
ncomp = 12
ncarac = 2

# LDA para visualizar
X = F
y = np.array([i for i in py.flatten([[i]*20 for i in range(12)])])
target_names = np.array(conf.artistas)

lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
principais = lda.coef_
princ_orig = principais

Fs = []
Prots = []
for i in range(12):
    Fs.append(X_r2[y == i])
    prot = np.array([np.mean(X_r2[y==i, k]) for k in range(X_r2.shape[1])])
    prot = np.nan_to_num(prot)
    Prots.append(prot)

# ordenamos a tabela pela ordem correta cronológica dos pintores
Ford = [Fs[conf.ordem[i]] for i in range(12)]
Prots = [Prots[conf.ordem[i]] for i in range(12)]
principais = np.array(Ford)


print 'COMPONENTES', principais
#print 'INDICE COMPONENTES', args
print 'CONTRIBUICAO', lda.scalings_

# plotamos os Prots, pinturas, dialéticas e série temporal
plt.figure(figsize=(15,15))
ax = plt.subplot(111)
aat = np.zeros(2)
aaf = np.zeros(2)

for i in xrange(principais.shape[0]):
    cc = np.zeros(3) + float(i) / principais.shape[0]
    x = principais[i, 0]
    y = principais[i, 1]
    aaf = np.sum(principais[:i+1], 0) / (i+1)

    ax.plot(aaf[0], aaf[1], 'o', color="#666666")
    if i != 0:
        ax.plot((aat[0], aaf[0]), (aat[1], aaf[1]), ':', color='#333333')
    aat = np.copy(aaf)

    ax.plot(x, y, 'bo')
    ax.text(x, y, str(i+1) + ' ' + agents[i], fontsize=11)

    # plotamos também as pinturas todas
    #ax.plot(Ford[i][:,0], Ford[i][:,1], 'o', label=conf.artistas[conf.ordem[i]],
    #            color=py.cm.jet(np.float(i) / (len(Ford)+1)), alpha=.4)
    # plotamos o protótipo (ponto médio)
    #prot = [np.mean(Ford[i][:,0]), np.mean(Ford[i][:,1])]
    #ax.plot(prot[0], prot[1], 'k+')

ax.plot(principais[:,0], principais[:,1], color="#000000")
plt.legend()
plt.savefig('caso2_g1.png')

# plt.clf()
# ax = plt.subplot(111)
# for i in xrange(dados.shape[0]):
#     cc = np.zeros(3) + float(i) / dados.shape[0]
#     x = dados[i, 0]
#     y = dados[i, 1]
#     aaf = np.sum(dados[:i+1], 0) / (i+1)

#     ax.plot(aaf[0], aaf[1], 'o', color="#666666")
#     if i != 0:
#         ax.plot((aat[0], aaf[0]), (aat[1], aaf[1]), ':', color='#333333')
#     aat = np.copy(aaf)
#     ax.plot(x, y, 'bo', label=agents[i])
#     #p.text(x, y, str(i+1), fontsize=12)
#     ax.text(x, y, str(i+1) + ' ' + agents[i], fontsize=11)

# ax.plot(dados[:,0], dados[:,1], color="#000000")
# plt.savefig('caso1_g2.png')

#
# Oposição e Inovação
#

princ_orig = dados
# para todos
oposicao=[]
inovacao=[]
for i in xrange(1, ncomp):
    a=princ_orig[i-1]    # conforme no artigo... a eh vi
    b=np.sum(princ_orig[:i+1],0)/(i+1) # meio   ... b eh a (average state)
    c=princ_orig[i] # ... c eh um vj

    Di=2*(b-a) # ... Di = 2 * a - vi
    Mij=c-a # ... Mij = vj - vi

    opos=np.sum(Di*Mij)/np.sum(Di**2)  # ... Wij = < Mij , Di > / || Di || ^ 2
    oposicao.append(opos)

    ########## Cálculo de inovação ##################
    # http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    inov=np.sqrt(  ( np.sum((a-c)**2)*np.sum((b-a)**2) - np.sum( (a-c)*(b-a) )**2 )/np.sum((b-a)**2)  )

    inovacao.append(inov)

#
# Dialética
#

dialeticas=[]

for i in xrange(2, ncomp):
   a=princ_orig[i-2] # thesis
   b=princ_orig[i-1] # antithesis
   c=princ_orig[i]   # synthesis

   # cálculo da dialética
   t1 = np.sum((b-a)*c)
   t2 = np.sum(-((b**2 - a**2)/2))
   t3 = np.sum((b-a)**2)
   dist = np.abs(t1 + t2) / np.sqrt(t3)

   dialeticas.append(dist)


print '\n###TABLE VII. TABLE VIII###.\n'
print '\n*** Oposição:\n', oposicao
print '\n*** Inovação:\n', inovacao
print '\n*** Dialética:\n', dialeticas

# plotando opos, inov e dial
fig = plt.figure(figsize=(13,12))
ax = fig.add_subplot(111)
ax.plot(range(len(oposicao)), oposicao, label="Oposicao")
for i in range(len(oposicao)):
    ax.text(i, oposicao[i], '%.2f' % oposicao[i], fontsize=11)
ax.plot(range(len(inovacao)), inovacao, label="Inovacao")
for i in range(len(inovacao)):
    ax.text(i, inovacao[i], '%.2f' % inovacao[i], fontsize=11)
plt.xticks(range(len(inovacao)), [r'Caravaggio $\rightarrow$ Frans Hals',
                                  r'Frans Hals $\rightarrow$ Poussin',
                                  r'Poussin $\rightarrow$ Velazquez',
                                  r'Velazquez $\rightarrow$ Rembrandt',
                                  r'Rembrandt $\rightarrow$ Vermeer',
                                  r'Vermeer $\rightarrow$ Van Gogh',
                                  r'Van Gogh $\rightarrow$ Kandinsky',
                                  r'Kandinsky $\rightarrow$ Matisse',
                                  r'Matisse $\rightarrow$ Picasso',
                                  r'Picasso $\rightarrow$ Miro',
                                  r'Miro $\rightarrow$ Pollock'])
fig.autofmt_xdate()
#ax.set_yticklabels([])
plt.legend()
plt.savefig("caso2_oposEinov.png")

plt.clf()
ax = fig.add_subplot(111)
ax.plot(range(len(dialeticas)), dialeticas, label="Dialetica")
for i in range(len(dialeticas)):
    ax.text(i, dialeticas[i], '%.2f' % dialeticas[i], fontsize=11)

dialabels = [r'Caravaggio $\rightarrow$ Frans Hals $\rightarrow$ Poussin',
             r'Frans Hals $\rightarrow$ Poussin $\rightarrow$ Velazquez',
             r'Poussin $\rightarrow$ Velazquez $\rightarrow$ Rembrandt',
             r'Velazquez $\rightarrow$ Rembrandt $\rightarrow$ Vermeer',
             r'Rembrandt $\rightarrow$ Vermeer $\rightarrow$ Van Gogh',
             r'Vermeer $\rightarrow$ Van Gogh $\rightarrow$ Kandinsky',
             r'Van Gogh $\rightarrow$ Kandinsky $\rightarrow$ Matisse',
             r'Kandinsky $\rightarrow$ Matisse $\rightarrow$ Picasso',
             r'Matisse $\rightarrow$ Picasso $\rightarrow$ Miro',
             r'Picasso $\rightarrow$ Miro $\rightarrow$ Pollock']

plt.xticks(range(len(dialeticas)), dialabels)
fig.autofmt_xdate()
plt.legend()
plt.savefig("caso2_dialetica.png")

# #
# # Perturbação
# #

# nperturb = 1000
# # distancias[original, ruido, amostra]
# distancias = np.zeros((ncomp, ncomp, nperturb))
# autovals = np.zeros((nperturb, 2))  # agora para 8d
# princ_orig = princ_orig[:,:2]
# #princ = princ[:,:2]

# for foobar in xrange(nperturb):
#     dist = np.random.randint(-2, 3, copia_dados.shape)
#     copia_dados += dist

#     for i in xrange(copia_dados.shape[1]):
#         copia_dados[:,i] = (copia_dados[:,i] - copia_dados[:,i].mean())/copia_dados[:,i].std()

#     # fazemos pca para dados considerando esses pontos aleatórios entre -2 e 2
#     # FIXME: substituir depois pca_nipals
#     T, P, E = pca.PCA_nipals(copia_dados)
#     autovals[foobar] = E[:2]
#     princ = T[:,:2]
#     for i in xrange(ncomp):
#         for j in xrange(ncomp):
#             distancias[i, j, foobar] = np.sum((princ_orig[i] - princ[j])**2)**.5

# stds = np.zeros((ncomp, ncomp))
# means = np.zeros((ncomp, ncomp))
# main_stds = []
# main_means = []
# print 'dados', copia_dados
# for i in xrange(ncomp):
#     for j in xrange(ncomp):
#         stds[i,j] = distancias[i,j,:].std()
#         means[i,j] = distancias[i,j,:].mean()
#         if i == j:
#           main_stds.append(stds[i,j])
#           main_means.append(means[i,j])
# np.savetxt("mean2_.txt",means,"%.2e")
# np.savetxt("stds2_.txt",stds,"%.2e")

# print '###TABLE V.### Average and standard deviation of the deviations for each composer and for the 8 eigenvalues.'

# print 'main_means', main_means
# print 'main_stds', main_stds

# # Cálculo das médias e variâncias dos desvios dos primeiros 4 autovalores

# deltas = autovals - autovalores_prop[:8]
# medias = deltas.mean(0)
# desvios = deltas.std(0)
# print 'eigenvalues means', medias
# print 'eigenvalues stds', desvios

