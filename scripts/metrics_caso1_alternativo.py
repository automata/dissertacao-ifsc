# encoding: utf-8

# CASO 1: sem PCA, 2 melhores features

# 1. considero apenas 2 melhores features
# 2. separo as pinturas em 12 classes
# 3. ordeno as pinturas em ordem cronológica
# 4. calculo os protótipos
# 5. calculo dialética, inovação e oposição dos Prots

import pickle as pk
import numpy as np
import pca as pc
import pylab as py
import config as conf
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as image

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
print F__.shape
# selecionamos as features
F__ = F__[:,[83,87]]
# normalizamos
M = np.mean(F__, axis=0)
D = np.std(F__, axis=0)
F = (F__-M) / D
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

# dados usados para cálculo dos 'metrics'
agents = [conf.artistas[conf.ordem[i]] for i in range(12)]
agents_orig = agents[:]
print agents_orig
dados = np.array(Prots)
ncomp = 12
ncarac = 2

# plotamos os Prots, pinturas, dialéticas e série temporal
fig = plt.figure(figsize=(30,30), dpi=72)
ax = plt.subplot(111)
aat = np.zeros(2)
aaf = np.zeros(2)

# arrumando van gogh e frans hals
agents[1] = 'Frans Hals'
agents[6] = 'van Gogh'
annotate_xy = [(-70,-50), (20,-50), (80,-30), (-50,-55), (-50,6), (-20,25),
               (15,50), (-40,50), (-50,-30), (-50,60), (-40,40), (-80,40)]

for i in xrange(dados.shape[0]):
    # cc = np.zeros(3) + float(i) / dados.shape[0]
    # x = dados[i, 0]
    # y = dados[i, 1]
    # aaf = np.sum(dados[:i+1], 0) / (i+1)

    # # ax.plot(aaf[0], aaf[1], 'o', color="#666666")
    # # if i != 0:
    # #     ax.plot((aat[0], aaf[0]), (aat[1], aaf[1]), ':', color='#333333')
    # # aat = np.copy(aaf)

    # ax.plot(x, y, 'bo', markersize=8)
    # if i == 1:
    #     ax.text(x+.05, y-.05, str(i+1) + ' ' + agents[i], fontsize=14)
    # elif i == 3:
    #     ax.annotate(str(i+1) + ': ' + agents[i], xy=(x,y),
    #                 xytext=annotate_xy[i], 
    #                 textcoords='offset points', ha='center', va='bottom',
    #                 arrowprops=dict(arrowstyle='-|>',
    #                                 connectionstyle='arc3,rad=0.1',
    #                                 color='red'), fontsize=14)
    # elif i == 4:
    #     ax.text(x-.4, y+.025, str(i+1) + ' ' + agents[i], fontsize=14)
    # elif i == 5:
    #     ax.text(x-.3, y+.05, str(i+1) + ' ' + agents[i], fontsize=14)
    # elif i == 7:
    #     ax.text(x-.5, y+.025, str(i+1) + ' ' + agents[i], fontsize=14)
    # elif i == 8:
    #     ax.text(x-.4, y-.025, str(i+1) + ' ' + agents[i], fontsize=14)
    # elif i == 10:
    #     ax.text(x+.05, y-.025, str(i+1) + ' ' + agents[i], fontsize=14)
    # elif i == 11:
    #     ax.text(x-.5, y+.025, str(i+1) + ' ' + agents[i], fontsize=14)
    # else:
    #     ax.text(x+.025, y+.025, str(i+1) + ' ' + agents[i], fontsize=14)

    # plotamos também as pinturas todas
    dpi = 72
    imageSize = (40,40)
    xs = Ford[i][:,0]
    ys = Ford[i][:,1]

    line, = ax.plot(xs, ys, "bo", mfc="None", mec="None", markersize=imageSize[0] * (dpi/ 96))
    ax.get_frame().set_alpha(0)
    ax.set_xlim((-1.5,2.0))
    ax.set_ylim((-0.7,3.0))
    line._transform_path()
    pat, affine = line._transformed_path.get_transformed_points_and_affine()
    pat = affine.transform_path(pat)
    j = 0
    path = agents_orig[i].lower()
    for pixelPoint in pat.vertices:
        # place image at point, centering it
        im = image.imread('pinturas/%s/thumb.r.%s.jpg.png' % (path,j))
        fig.figimage(im,pixelPoint[0]-imageSize[0]/2,pixelPoint[1]-imageSize[1]/2,origin="upper")
        j += 1

    # ax.plot(Ford[i][:,0], Ford[i][:,1], 'o', label=str(i+1) + ': ' + agents[i],
    #         color=py.cm.jet(np.float(i) / (len(Ford)+1)), alpha=.4)
    # # plotamos o protótipo (ponto médio)
    # prot = [np.mean(Ford[i][:,0]), np.mean(Ford[i][:,1])]
    # ax.plot(prot[0], prot[1], 'k+')

#ax.plot(dados[:,0], dados[:,1], color="#000000")
#plt.xlim((-2,2.5))
#plt.ylim((-1,4))
plt.xlabel(r'$\mu$ of curvature pikes', fontsize=14)
plt.ylabel(r'$\mu$ of number of segments', fontsize=14)
#plt.legend(loc='upper left')
#plt.title(r'a) Projected "creative space"')
plt.savefig('caso1_g1_alternativo.pdf', bbox_inches='tight')

# plt.clf()
# ax = plt.subplot(111)
# for i in xrange(dados.shape[0]):
#     cc = np.zeros(3) + float(i) / dados.shape[0]
#     x = dados[i, 0]
#     y = dados[i, 1]
#     aaf = np.sum(dados[:i+1], 0) / (i+1)

#     # ax.plot(aaf[0], aaf[1], 'o', color="#666666")
#     # if i != 0:
#     #     ax.plot((aat[0], aaf[0]), (aat[1], aaf[1]), ':', color='#333333')
#     # aat = np.copy(aaf)

#     ax.plot(x, y, 'bo', markersize=8)
#     #ax.text(x, y, str(i+1) + ' ' + agents[i], fontsize=11)

#     ax.annotate(str(i+1) + ': ' + agents[i], xy=(x,y), xytext=annotate_xy[i], 
#                 textcoords='offset points', ha='center', va='bottom',
#                 arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=0.1',
#                                 color='red'), fontsize=14)
    

#     # plotamos também as pinturas todas
#     # ax.plot(Ford[i][:,0], Ford[i][:,1], 'o', label=str(i+1) + ': ' + agents[i],
#     #         color=py.cm.jet(np.float(i) / (len(Ford)+1)), alpha=.4)
#     # # plotamos o protótipo (ponto médio)
#     # prot = [np.mean(Ford[i][:,0]), np.mean(Ford[i][:,1])]
#     # ax.plot(prot[0], prot[1], 'k+')

# ax.plot(dados[:,0], dados[:,1], color="#000000")
# plt.xlim((-2,2.5))
# plt.ylim((-1,4))
# plt.xlabel(r'$\mu$ of curvature pikes', fontsize=14)
# plt.ylabel(r'$\mu$ of number of segments', fontsize=14)
# #plt.legend(loc='upper left')
# #plt.title(r'b) Time-series')
# plt.savefig('caso1_g2.pdf', bbox_inches='tight')

# #
# # Oposição e Inovação
# #

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
# dialeticas2d=[]
# dialeticasTodos=[]
# print 'ncomp', ncomp
# for i in xrange(2, ncomp):
#    a=princ_orig[i-2] # thesis
#    b=princ_orig[i-1] # antithesis
#    c=princ_orig[i]   # synthesis

#    # dialetica para 2d
#    dist2d = np.abs( (b[0]-a[0])*c[0] + (b[1]-a[1])*c[1] +
#                  (-((b[0]**2 - a[0]**2)/2)
#                   -((b[1]**2 - a[1]**2)/2)) ) / np.sqrt( (b[0]-a[0])**2 + (b[1]-a[1])**2)

#    dialeticas2d.append(dist2d)


# print '\n###TABLE VII. TABLE VIII###.\n'
# print '\n*** Oposição:\n', oposicao
# print '\n*** Inovação:\n', inovacao
# #print '\n*** Dialéticas:\n', dialeticas
# print '\n*** Dialéticas 2d:\n', dialeticas2d
# #print '\n*** Dialéticas Todos (8d):\n', dialeticasTodos
# #dialeticas = np.array(np.abs(dialeticas))
# #print ( (dialeticas-dialeticas.min())/(dialeticas.max()-dialeticas.min()) )

# # plotando opos, inov e dial
# fig = plt.figure(figsize=(13,12))
# ax = fig.add_subplot(111)
# ax.plot(range(len(oposicao)), oposicao, label="Opposition")
# ax.plot(range(len(oposicao)), oposicao, 'bo')
# for i in range(len(oposicao)):
#     if i != (len(oposicao)-1):
#         ax.text(i, oposicao[i], '%.2f' % oposicao[i], fontsize=14)
#     else:
#         ax.text(i-.5, oposicao[i], '%.2f' % oposicao[i], fontsize=14)
# ax.plot(range(len(inovacao)), inovacao, label="Skewness")
# ax.plot(range(len(inovacao)), inovacao, 'g^')
# for i in range(len(inovacao)):
#     if i != (len(oposicao)-1):
#         ax.text(i, inovacao[i], '%.2f' % inovacao[i], fontsize=14)
#     else:
#         ax.text(i-.5, inovacao[i], '%.2f' % inovacao[i], fontsize=14)
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
#                                   r'Miro $\rightarrow$ Pollock'], fontsize=14)
# fig.autofmt_xdate()
# #ax.set_yticklabels([])
# plt.legend(loc='upper left')
# plt.savefig("caso1_oposEinov.pdf", bbox_inches='tight')

# plt.clf()
# ax = fig.add_subplot(111)
# ax.plot(range(len(dialeticas2d)), dialeticas2d, 'r', label="Counter-dialectics")
# ax.plot(range(len(dialeticas2d)), dialeticas2d, 'ro')
# for i in range(len(dialeticas2d)):
#     if i != len(dialeticas2d)-1:
#         ax.text(i, dialeticas2d[i], '%.2f' % dialeticas2d[i], fontsize=14)
#     else:
#         ax.text(i-.5, dialeticas2d[i], '%.2f' % dialeticas2d[i], fontsize=14)

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

# plt.xticks(range(len(dialeticas2d)), dialabels, fontsize=14)
# fig.autofmt_xdate()
# plt.legend(loc='upper left')
# plt.savefig("caso1_dialetica.pdf",  bbox_inches='tight')

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

