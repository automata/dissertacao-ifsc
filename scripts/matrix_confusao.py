# encoding: utf-8

# CASO 3: LDA usado para classificar

# 1. normalizo F
# 2. separo as pinturas em 12 classes
# 3. calculo LDA
# 4. calculo os protótipos a partir do LDA (Prots)
# 5. calculo dialética, inovação e oposição dos Prots

import pickle as pk
import numpy as np
import pca as pc
import pylab as py
import config as conf
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as image
import random

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix

# matriz de dados (feature matrix)
f = open('../ana-pintores/dados/feature_matrix1.pkl', 'rb')
Fa = np.array(pk.load(f))
f.close()

f = open('../ana-pintores/dados/feature_matrix2.pkl', 'rb')
Fb = np.array(pk.load(f))
f.close()

F = np.concatenate((Fa, Fb), axis=1)

# filtramos a tabela de pinturas, retirando colunas com valores zerados e
# degenerados
cols = [0,1,2] + range(5,12) + range(14,21) + range(23,30) + range(32,51) + range(52,64) + range(65,77) + range(78,90) + range(91,97)+ range(98,103)
F__ = F[:,cols]

# ordenamos os pintores antes...
F__ord = np.vstack([F__[(i*20):(i*20)+20] for i in conf.ordem])

# vamos criar um conjunto de treino, com 10 primeiras pinturas de cada artista
#Ftreino = np.vstack([F__[i:i+20][:10] for i in range(0, 12*20, 20)])

# calculamos a matriz de confusão N vezes
N = 100
cms = []
for i in range(N):
    rind = random.sample(range(20), 20)
    Ftreino = np.vstack([F__ord[i:i+20][rind[:10]] for i in range(0, 12*20, 20)])
    # e um conjunto de teste, com 10 últimas pinturas de cada artista
    Fteste = np.vstack([F__ord[i:i+20][rind[10:]] for i in range(0, 12*20, 20)])
    # normalizamos todos
    Mtreino = np.mean(Ftreino, axis=0)
    Dtreino = np.std(Ftreino, axis=0)
    Ftreino = np.nan_to_num((Ftreino-Mtreino) / Dtreino)
    
    Mteste = np.mean(Fteste, axis=0)
    Dteste = np.std(Fteste, axis=0)
    Fteste = np.nan_to_num((Fteste-Mteste) / Dteste)
    
    # LDA
    
    Xtreino = Ftreino
    Xteste = Fteste
    y = np.array([i for i in py.flatten([[i]*10 for i in range(12)])])
    target_names = np.array(conf.artistas)
    
    # aplicamos LDA no conjunto de treino e teste (após fitar... treinar com o
    # conjunto de treino)
    lda = LDA(n_components=2)
    # lda.fit(Xtreino, y, store_covariance=True)
    Xtreino_r2 = lda.fit(Xtreino, y, store_covariance=True).transform(Xtreino)
    
    y_pred = lda.predict(Xteste)
    print y_pred
    cm = confusion_matrix(y, y_pred)
    cms.append(cm)
    print 'cm', cm

cm_media = sum([np.array(cm, dtype=float) for cm in cms]) / N
print cm_media
fig = plt.figure()
ax = plt.subplot(111)
cax = ax.matshow(cm_media, interpolation='nearest', cmap=py.cm.jet)
#py.title('Confusion matrix')
plt.colorbar(cax)
plt.ylabel('True paintings', fontsize=11)
plt.xlabel('Predicted paintings', fontsize=11)
dialabels = [r'Caravaggio',
             r'Frans Hals',
             r'Poussin',
             r'Velazquez',
             r'Rembrandt',
             r'Vermeer',
             r'Van Gogh',
             r'Kandinsky',
             r'Matisse',
             r'Picasso',
             r'Miro',
             r'Pollock']

plt.yticks(range(len(dialabels)), dialabels, fontsize=11)
plt.xticks(range(len(dialabels)), dialabels, rotation='vertical', fontsize=11)
#plt.show()
fig.savefig('matriz_confusao.pdf', bbox_inches='tight')

# #print 'S,V,X::', lda.S, lda.V, lda.Xorig
# print '*** treino ***'
# autoval,autovet = lda.S, lda.V.T
# autoval_prop = autoval / np.sum(autoval)
# print 'autovalores', autoval / autoval.max()
# print 'autovetores', autovet
# print 'autovalores prop.', [float(i) for i in autoval_prop]

# print 'dados', Xtreino_r2
# print 'params:', lda.get_params()
# print 'coeficientes:', lda.coef_.shape
# print 'medias:', lda.means_
# print 'priors:', lda.priors_
# print 'scalings:', lda.scalings_
# print 'covar:', lda.covariance_.shape
# print 'contrib autovals:', lda.coef_ / np.sum(lda.coef_)
# print 'a', lda.coef_.shape, lda.covariance_.shape

# Xteste_r2 = lda.fit(Xtreino, y, store_covariance=True).transform(Xteste)
# #Xteste_r2 = lda.transform(Xteste)
# print '*** teste ***'
# autoval,autovet = lda.S, lda.V.T
# autoval_prop = autoval / np.sum(autoval)
# print 'autovalores', autoval
# print 'autovetores', autovet
# print 'autovalores prop.', autoval_prop

# print 'dados', Xteste_r2
# print 'params:', lda.get_params()
# print 'coeficientes:', lda.coef_.shape
# print 'medias:', lda.means_
# print 'priors:', lda.priors_
# print 'scalings:', lda.scalings_
# print 'covar:', lda.covariance_.shape
# print 'contrib autovals:', lda.coef_ / np.sum(lda.coef_)
# print 'a', lda.coef_.shape, lda.covariance_.shape
# print 'shape treino', Xtreino_r2.shape
# print 'shape teste', Xteste_r2.shape
# # c1 = lda.coef_[:,0]
# # c1 / sum(abs(c1)) * 100
# # print 'c1', c1

# # autovalores, autovetores = np.linalg.eig(lda.covariance_)
# # autovalores_desord = autovalores.copy()
# # args = np.argsort(lda.coef_)[::-1]
# # autovalores = autovalores[args]
# # autovetores = autovetores[args]

# # # calculamos os componentes principais para todos os dados
# # dados_finais_ = np.dot(autovetores.T, F.T)
# # principais_ = dados_finais_.T

# # # proporção dos autovalores
# # autovalores_prop = autovalores / np.sum(autovalores)
# # print 'contrib autovals:', np.around(autovalores_prop, decimals=2)
# # print 'args:', args
# #print X_r2
# # separamos as pinturas em classes de 20 pinturas, uma para cada pintor
# # e calculamos os protótipos (pontos médios)

# print 'TREINO', [(i, Xtreino_r2[y == i]) for i in range(12)]

# print 'TESTE', [(i, Xteste_r2[y == i]) for i in range(12)]

# ######## TREINO
# X_r2 = Xtreino_r2

# Fs = []
# Prots = []
# for i in range(12):
#     Fs.append(X_r2[y == i])
#     prot = np.array([np.mean(X_r2[y == i, k]) for k in range(X_r2.shape[1])])
#     prot = np.nan_to_num(prot)
#     Prots.append(prot)

# agents = [conf.artistasD[conf.ordem[i]] for i in range(12)]

# # ordenamos a tabela pela ordem correta cronológica dos pintores
# Ford = [Fs[conf.ordem[i]] for i in range(12)]
# Prots = [Prots[conf.ordem[i]] for i in range(12)]
# principais = np.array(Ford)
# print Ford
# print 'PRINCIPAIS 1:', principais[0]
# #print 'PRINCIPAIS 2:', principais_.T[0]

# agents[1] = 'Frans Hals'
# agents[6] = 'van Gogh'
# annotate_xy = [(-70,-50), (20,-50), (90,-40), (-50,-55), (110,-20), (-50,10),
#                (15,50), (-60,30), (-100,-50), (-30,70), (-40,40), (-80,40)]

# plt.figure(figsize=(12,12))
# ax = plt.subplot(111)
# for i in range(12):
#     x = -Prots[i][1]
#     y = Prots[i][0]
#     aaf = np.sum(Prots[:i+1], 0) / (i+1)
#     ax.plot(-aaf[1], aaf[0], 'o', color="#666666")
#     if i != 0:
#         ax.plot((-aat[1], -aaf[1]), (aat[0], aaf[0]), ':', color='#333333')
#     aat = np.copy(aaf)
#     ax.plot(x, y, 'bo')
#     #ax.text(x, y, str(i+1) + ': ' + conf.artistas[conf.ordem[i]], fontsize=11)

#     ax.annotate(str(i+1) + ': ' + agents[i], xy=(x,y), xytext=annotate_xy[i], 
#                 textcoords='offset points', ha='center', va='bottom',
#                 arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=0.1',
#                                 color='red'), fontsize=14)


#     # plotamos também as pinturas todas
#     ax.plot(Ford[i][:,1], Ford[i][:,0], 'o',
#             label=str(i+1) + ': ' + agents[i],
#             color=py.cm.jet(np.float(i) / (len(Ford)+1)), alpha=.4)
#     # plotamos o protótipo (ponto médio)
#     ax.plot(-Prots[i][1], Prots[i][0], 'k+')
# Prots = np.array(Prots)
# ax.plot(-Prots[:,1], Prots[:,0], c='#000000')
# plt.legend(loc='upper left')
# plt.ylabel('First Component')
# plt.xlabel('Second Component')
# #plt.title('LDA')
# #plt.savefig('valida3_g1_treino.pdf', bbox_inches='tight')




# # dados usados para cálculo dos 'metrics'
# agents = [conf.artistas[conf.ordem[i]] for i in range(12)]
# dados = np.array(Prots)
# ncomp = 12
# ncarac = 2

# #
# # Oposição e Inovação
# #

# for i in xrange(dados.shape[1]):
#     dados[:,i] = (dados[:,i] - dados[:,i].mean())/dados[:,i].std()

# princ_orig = dados
# # para todos
# oposicao=[]
# inovacao=[]
# for i in xrange(1, ncomp):
#     a=princ_orig[i-1]    # conforme no artigo... a eh vi
#     b=np.sum(princ_orig[:i+1],0)/(i+1) # meio   ... b eh a (average state)
#     c=princ_orig[i] # ... c eh um vj

#     Di=2*(b-a) # ... Di = 2 * a - vi
#     Mij=c-a # ... Mij = vj - vi

#     opos=np.sum(Di*Mij)/np.sum(Di**2)  # ... Wij = < Mij , Di > / || Di || ^ 2
#     oposicao.append(opos)

#     ########## Cálculo de inovação ##################
#     # http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
#     inov=np.sqrt(  ( np.sum((a-c)**2)*np.sum((b-a)**2) - np.sum( (a-c)*(b-a) )**2 )/np.sum((b-a)**2)  )

#     inovacao.append(inov)

# #
# # Dialética
# #

# dialeticas=[]

# for i in xrange(2, ncomp):
#    a=princ_orig[i-2] # thesis
#    b=princ_orig[i-1] # antithesis
#    c=princ_orig[i]   # synthesis

#    # cálculo da dialética
#    t1 = np.sum((b-a)*c)
#    t2 = np.sum(-((b**2 - a**2)/2))
#    t3 = np.sum((b-a)**2)
#    dist = np.abs(t1 + t2) / np.sqrt(t3)

#    dialeticas.append(dist)


# print '\n###TABLE VII. TABLE VIII###.\n'
# print '\n*** Oposição:\n', oposicao
# print '\n*** Inovação:\n', inovacao
# print '\n*** Dialética:\n', dialeticas

# oposicao = np.nan_to_num(oposicao)
# inovacao = np.nan_to_num(inovacao)
# dialeticasa = np.nan_to_num(dialeticas)

# # plotando opos, inov e dial
# fig = plt.figure(figsize=(13,12))
# ax = fig.add_subplot(111)
# ax.plot(range(len(oposicao)), oposicao, label="Opposition")
# for i in range(len(oposicao)):
#     ax.text(i, oposicao[i], '%.2f' % oposicao[i], fontsize=11)
# ax.plot(range(len(inovacao)), inovacao, label="Skewness")
# for i in range(len(inovacao)):
#     ax.text(i, inovacao[i], '%.2f' % inovacao[i], fontsize=11)
# plt.xticks(range(len(inovacao)), [r'Caravaggio $\rightarrow$ Frans Hals',
#                                   r'Frans Hals $\rightarrow$ Poussin',
#                                   r'Poussin $\rightarrow$ Velazquez',
#                                   r'Velazquez $\rightarrow$ Rembrandt',
#                                   r'Rembrandt $\rightarrow$ Vermeer',
#                                   r'Vermeer $\rightarrow$ Van Gogh',
#                                   r'Van Gogh $\rightarrow$ Kandinsky',
#                                   r'Kandinsky $\rightarrow$ Matisse',
#                                   r'Matisse $\rightarrow$ Picasso',
#                                   r'Picasso $\rightarrow$ Miro',
#                                   r'Miro $\rightarrow$ Pollock'])
# fig.autofmt_xdate()
# #ax.set_yticklabels([])
# plt.legend()
# #plt.savefig("valida3_oposEinov_treino.pdf", bbox_inches='tight')

# plt.clf()
# ax = fig.add_subplot(111)
# ax.plot(range(len(dialeticas)), dialeticas, label="Counter-dialectics")
# for i in range(len(dialeticas)):
#     ax.text(i, dialeticas[i], '%.2f' % dialeticas[i], fontsize=11)

# dialabels = [r'Caravaggio $\rightarrow$ Frans Hals $\rightarrow$ Poussin',
#              r'Frans Hals $\rightarrow$ Poussin $\rightarrow$ Velazquez',
#              r'Poussin $\rightarrow$ Velazquez $\rightarrow$ Rembrandt',
#              r'Velazquez $\rightarrow$ Rembrandt $\rightarrow$ Vermeer',
#              r'Rembrandt $\rightarrow$ Vermeer $\rightarrow$ Van Gogh',
#              r'Vermeer $\rightarrow$ Van Gogh $\rightarrow$ Kandinsky',
#              r'Van Gogh $\rightarrow$ Kandinsky $\rightarrow$ Matisse',
#              r'Kandinsky $\rightarrow$ Matisse $\rightarrow$ Picasso',
#              r'Matisse $\rightarrow$ Picasso $\rightarrow$ Miro',
#              r'Picasso $\rightarrow$ Miro $\rightarrow$ Pollock']

# plt.xticks(range(len(dialeticas)), dialabels)
# fig.autofmt_xdate()
# plt.legend()
# #plt.savefig("valida3_dialetica_treino.pdf", bbox_inches='tight')

# # #
# # # Perturbação
# # #

# # nperturb = 1000
# # # distancias[original, ruido, amostra]
# # distancias = np.zeros((ncomp, ncomp, nperturb))
# # autovals = np.zeros((nperturb, 2))  # agora para 8d
# # princ_orig = princ_orig[:,:2]
# # #princ = princ[:,:2]

# # for foobar in xrange(nperturb):
# #     dist = np.random.randint(-2, 3, copia_dados.shape)
# #     copia_dados += dist

# #     for i in xrange(copia_dados.shape[1]):
# #         copia_dados[:,i] = (copia_dados[:,i] - copia_dados[:,i].mean())/copia_dados[:,i].std()

# #     # fazemos pca para dados considerando esses pontos aleatórios entre -2 e 2
# #     # FIXME: substituir depois pca_nipals
# #     T, P, E = pca.PCA_nipals(copia_dados)
# #     autovals[foobar] = E[:2]
# #     princ = T[:,:2]
# #     for i in xrange(ncomp):
# #         for j in xrange(ncomp):
# #             distancias[i, j, foobar] = np.sum((princ_orig[i] - princ[j])**2)**.5

# # stds = np.zeros((ncomp, ncomp))
# # means = np.zeros((ncomp, ncomp))
# # main_stds = []
# # main_means = []
# # print 'dados', copia_dados
# # for i in xrange(ncomp):
# #     for j in xrange(ncomp):
# #         stds[i,j] = distancias[i,j,:].std()
# #         means[i,j] = distancias[i,j,:].mean()
# #         if i == j:
# #           main_stds.append(stds[i,j])
# #           main_means.append(means[i,j])
# # np.savetxt("mean2_.txt",means,"%.2e")
# # np.savetxt("stds2_.txt",stds,"%.2e")

# # print '###TABLE V.### Average and standard deviation of the deviations for each composer and for the 8 eigenvalues.'

# # print 'main_means', main_means
# # print 'main_stds', main_stds

# # # Cálculo das médias e variâncias dos desvios dos primeiros 4 autovalores

# # deltas = autovals - autovalores_prop[:8]
# # medias = deltas.mean(0)
# # desvios = deltas.std(0)
# # print 'eigenvalues means', medias
# # print 'eigenvalues stds', desvios

