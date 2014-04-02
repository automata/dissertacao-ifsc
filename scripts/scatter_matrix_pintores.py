# encoding: utf-8

import pickle as pk
import numpy as np
import pca as pc
import pylab as py
import config as conf
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as image

# 0.   entropia da imagem
# 1.   média das energias das linhas da imagem
# 2.   desvio padrão das energias das linhas da imagem
#      média das energias das colunas da imagem
#      desvio padrão das energias das colunas da imagem
# 3.   centroide das energias das linhas
# 4.   centroide das energias das colunas
# 5.   média das energias das linhas e colunas (total)
# 6.   desvio padrão das energias das linhas e colunas (total)
# vermelho:
# 7.   entropia da imagem
# 8.  média das energias das linhas da imagem
# 9.  desvio padrão das energias das linhas da imagem
#      média das energias das colunas da imagem
#      desvio padrão das energias das colunas da imagem
# 10.  centroide das energias das linhas
# 11.  centroide das energias das colunas
# 12.  média das energias das linhas e colunas (total)
# 13.  desvio padrão das energias das linhas e colunas (total)
# verde:
# 14.  entropia da imagem
# 15.  média das energias das linhas da imagem
# 16.  desvio padrão das energias das linhas da imagem
#      média das energias das colunas da imagem
#      desvio padrão das energias das colunas da imagem
# 17.  centroide das energias das linhas
# 18.  centroide das energias das colunas
# 19.  média das energias das linhas e colunas (total)
# 20.  desvio padrão das energias das linhas e colunas (total)
# azul:
# 21.  entropia da imagem
# 22.  média das energias das linhas da imagem
# 23.  desvio padrão das energias das linhas da imagem
#      média das energias das colunas da imagem
#      desvio padrão das energias das colunas da imagem
# 24.  centroide das energias das linhas
# 25.  centroide das energias das colunas
# 26.  média das energias das linhas e colunas (total)
# 27.  desvio padrão das energias das linhas e colunas (total)
# cinza:
# 28.  média das entropias locais (disco 5x5)
# 29.  média das entropias locais (disco 50x50)
# 30.  quantidade de linhas (segundo transformada de Hough) da imagem
# haralick da cinza (im cinza => equalizada => filtro media)
# http://murphylab.web.cmu.edu/publications/boland/boland_node26.html
# haralick para direção de adjacência 1 --
# 31.  angular second moment
# 32.  contrast
# 33.  correlation
# 34.  sum of squares: variance
# 35.  inverse difference moment
# 36.  sum average
# 37.  sum variance
# 38.  sum entropy
# 39.  entropy
# 40.  difference average
# 41.  difference entropy
# 42.  info. measure of correlation 1
#      info. measure of correlation 2
# haralick para direção de adjacência 2 |
# 43.  angular second moment
# 44.  contrast
# 45.  correlation
# 46.  sum of squares: variance
# 47.  inverse difference moment
# 48.  sum average
# 49.  sum variance
# 50.  sum entropy
# 51.  entropy
# 52.  difference average
# 53.  difference entropy
# 54.  info. measure of correlation 1
#      info. measure of correlation 2
# haralick para direção de adjacência 2 \
# 55.  angular second moment
# 56.  contrast
# 57.  correlation
# 58.  sum of squares: variance
# 59.  inverse difference moment
# 60.  sum average
# 61.  sum variance
# 62.  sum entropy
# 63.  entropy
# 64.  difference average
# 65.  difference entropy
# 66.  info. measure of correlation 1
#      info. measure of correlation 2
# haralick para direção de adjacência 2 /
# 67.  angular second moment
# 68.  contrast
# 69.  correlation
# 70.  sum of squares: variance
# 71.  inverse difference moment
# 72.  sum average
# 73.  sum variance
# 74.  sum entropy
# 75.  entropy
# 76.  difference average
# 77.  difference entropy
# 78.  info. measure of correlation 1
#      info. measure of correlation 2
# para cada região conexa obtida a partir da segmentação SLIC (imagem binária):
# 79.  média das médias das distâncias (euclidiana) entre os picos
# 80.  média dos desvios padrão das distâncias (euclidiana) entre os picos
# 81.  média das médias das distâncias (em pixels do contorno) entre os picos
# 82.  média dos desvios padrão das distâncias (em pixels do contorno) entre os picos
# 83.  média das quantidades de picos da curvatura
# 84.  média dos perímetros das curvaturas
#      média das médias dos valores (na matriz) dos picos
# 85.  média das áreas dos segmentos
# 86.  média das razões perímetro**2 / área dos segmentos
# 87. média das quantidades de segmentos por pintura
# 88. média das áreas das regiões convexas (convex hull)
# 89. média das razões área convexa / área original (convex hull)

# matriz de dados (feature matrix)
f = open('dados/feature_matrix1.pkl', 'rb')
Fa = np.array(pk.load(f))
f.close()

f = open('dados/feature_matrix2.pkl', 'rb')
Fb = np.array(pk.load(f))
f.close()

F = np.concatenate((Fa, Fb), axis=1)

print 'F', F.shape

# selecionamos as colunas que queremos (pré-processamento para eleminar colunas
# com zeros ou NaNs)
#cols = range(0,51) + range(52,64) + range(65,77) + range(78,90) + [95,96] + range(98,103)
# sem medidas de curvatura:
#cols = [0,1,2] + range(5,12) + range(14,21) + range(23,30) + range(32,51) + range(52,64) + range(65,77) + range(78,90) + [95,96] + range(98,103)
# com:
cols = [0,1,2] + range(5,12) + range(14,21) + range(23,30) + range(32,51) + range(52,64) + range(65,77) + range(78,90) + range(91,97)+ range(98,103)
F__ = F[:,cols]
# normalizamos
M = np.mean(F__, axis=0)
D = np.std(F__, axis=0)
F = (F__-M) / D
print 'F'
# número de features
Nf = F.shape[1]
print 'número de features', Nf, Nf

### SCATTER MATRIX PARA PARES POSSIVEIS

# combinações de Nf elementos tomados 2 a 2, sem repetições e iguais
w = []
for x in itertools.product(range(Nf), range(Nf)):
    if (len(x) == len(set(x))) and (tuple(reversed(x)) not in w):
        w.append(x)

# analisamos alpha para cada combinação de NfxNf
alphas = []
for a,b in w:
    # número de classes diferentes
    Nc = 12
    # número de features (como temos pares a,b de features, é 2)
    Nf = 2
    # par de features a,b que iremos analisar
    F_ = F[:,[a,b]]
    # separamos em 12 classes, 20 pinturas para cada pintor
    Fs = [F_[i:i+20] for i in range(0,240,20)]
    
    # global mean feature vector
    # vetor contendo a média de todos os objetos
    M = np.mean(F_, axis=0).reshape((1,Nf))
    # global std feature vector
    D = np.std(F_, axis=0).reshape((1,Nf))
    
    # mean feature vector para cada classe
    Ms = [np.mean(Fs[i], axis=0).reshape((1,Nf)) for i in range(Nc)]
    # std feature vector para cada classe
    Ds = [np.std(Fs[i], axis=0).reshape((1,Nf)) for i in range(Nc)]
    
    # total scatter matrix
    S = ((F_-M).T).dot(F_-M)
    
    # scatter matrix para cada classe
    Ss = [((Fs[i]-Ms[i]).T).dot(Fs[i]-Ms[i]) for i in range(Nc)]
    
    # intraclass scatter matrix
    Sintra = np.array(sum(Ss))

    # quantidade de objetos (linhas) em cada classe
    Ns = [Fs[i].shape[0] for i in range(Nc)]
    # interclass scatter matrix
    Sinter = sum([Ns[i]*(((Ms[i]-M).T).dot(Ms[i]-M)) for i in range(Nc)])

    # validando... S == S_
    S_ = Sintra + Sinter
    Sintra = np.nan_to_num(Sintra)
    Sinter = np.nan_to_num(Sinter)
    S_ = np.nan_to_num(S_)
    # quantificando através de um funcional (traço) o quanto os grupos de
    # features estão inter e intra relacionados (proporção)
    if np.linalg.det(Sintra) != 0: # evitando matrizes singulares...
        ratio = Sinter.dot(np.linalg.inv(Sintra))
        alpha = ratio.trace()
        
        alphas.append([alpha, a, b, Fs]) 

# ordenamos os dados por alpha (menor é melhor)
alphas_sorted = sorted(alphas, key=lambda x:x[0], reverse=True)

# scatter plot considerando n pares de medidas, classes sobrepostas
n = 16

plt.figure(figsize=(13,12))
for i in xrange(n):
    alpha, a, b, Fs = alphas_sorted[i]
    Ford = [Fs[conf.ordem[j]] for j in range(12)]
    ax = plt.subplot(np.round(np.sqrt(n)), np.round(np.sqrt(n)),i+1)
    for j in xrange(len(Fs)):
        ax.plot(Ford[j][:,0], Ford[j][:,1], 'o',
                color=py.cm.jet(np.float(j) / 13), alpha=.4)
    plt.title(r'pair %s:  $\alpha$ = %.3f' % (i+1,alpha), fontsize=14)
    ax.set_xlim((-3,6))
    ax.set_ylim((-4,6))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #print '%i. alpha: %s. eixo-1: %s. eixo-2: %s.' % (i, alpha, a, b)

plt.savefig('sm_pintores.pdf', bbox_inches='tight')

# mostrando cada scatter plot, com labels, apenas 5 primeiros
n = 5
plt.figure(figsize=(13,12))
for i in xrange(n):
    plt.clf()
    alpha, a, b, Fs = alphas_sorted[i]
    ax = plt.subplot(111)
    for j in xrange(len(Fs)):
        if j >= 6:
            marker = 's'
        else:
            marker = 'o'
        ax.plot(Fs[j][:,0], Fs[j][:,1], marker, label=conf.artistas[j],
                color=py.cm.jet(np.float(j) / (len(Fs)+1)))
    plt.title('%s/%s %.3f' % (a,b,alpha), fontsize='small')
    #ax.set_xlim((-2,6))
    #ax.set_ylim((-4,5))
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #print '%i. alpha: %s. eixo-1: %s. eixo-2: %s.' % (i, alpha, a, b)
    plt.legend()
    plt.savefig('sm_pintores_par%s.png' % i)

