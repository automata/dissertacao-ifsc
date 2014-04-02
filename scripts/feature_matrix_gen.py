# encoding:utf-8

import pickle as pk
import numpy as np
import pca as pc
import pylab as p
import config as conf

# lendo features sobre entropias em uma tabela D
f = open('dados/pinturas.pkl', 'rb')
d_pinturas = pk.load(f)
f.close()

D = []
for i in d_pinturas:
    for j in i[1]:
        D.append(j)

# lendo features sobre curvaturas em uma tabela D2
d_art = {}
D2 = []
artistas = [x.lower() for x in conf.artistas]
for artista in artistas:
    print 'artista:', artista
    f = open('dados/segs_%s.pkl' % artista)
    d_segs, n_segs = pk.load(f)
    f.close()

    f_convexhull = open('dados/convexhull_%s.pkl' % artista)
    d_convexhull = pk.load(f_convexhull)
    f_convexhull.close()
    
    d_art[artista] = []
    i = 0
    t = 0
    for pint in xrange(conf.NUM_PINTURAS):
        print ' pintura:', pint
        for k in xrange(n_segs[i]):
            print '  segmento %s de %s.' % (k, n_segs[i])
            # nbr, C, peri, L, V, Aor, Ac, Rc, Pa = d_segs[t]
            nbr, C, peri, L, V, Aor, Pa = d_segs[t]
            Ac, Ca = d_convexhull[t]
            # calculamos distancias dos picos (euclidiana e pixels do perimet.)
            xs = np.array([np.real(C[L[j]]) for j in xrange(nbr)])
            ys = np.array([np.imag(C[L[j]]) for j in xrange(nbr)])
            d_eucli = [np.sqrt((xs[j+1]-ys[j+1])**2 + (xs[j]-ys[j])**2)
                       for j in xrange(nbr - 1)]
            d_pixel = [(L[j+1] - L[j])**2
                       for j in xrange(nbr - 1)]
            d_eucli = np.array(d_eucli)
            d_pixel = np.array(d_pixel)
            
            #print 'nbr, peri, V, Aor, Pa', nbr, peri, np.mean(V), Aor, Pa
            # agregamos todas as medidas em um vetor
            foo = [np.mean(d_eucli), np.std(d_eucli),
                   np.mean(d_pixel), np.std(d_pixel),
                   nbr, peri, np.mean(V), Aor, Pa, n_segs[i], Ac, Ca]
            d_art[artista].append(foo)
            t += 1
        # tiramos a media das medias de distancias entre picos
        # para poder incorporá-las à tabela D
        # bar terá dados de distancias de picos e outra metricas para CADA
        # SEGMENTO (REGIAO CONEXA)!
        # cada elemento em bar possui media dos valores das medidas em (foo)
        bar = [np.mean([x[0] for x in d_art[artista]]), # media/media de d_eucl
               np.mean([x[1] for x in d_art[artista]]), # media/std de d_eucli
               np.mean([x[2] for x in d_art[artista]]), # media/media de d_pix
               np.mean([x[3] for x in d_art[artista]]), # media/std de d_pixel
               np.mean([x[4] for x in d_art[artista]]), # media/qtd picos
               np.mean([x[5] for x in d_art[artista]]), # media/perimetro
               np.mean([x[6] for x in d_art[artista]]), # media/media val picos
               np.mean([x[7] for x in d_art[artista]]), # media/area do seg
               np.mean([x[8] for x in d_art[artista]]), # media/(per**2 / area)
               np.mean([x[9] for x in d_art[artista]]), # media/qtd segs 
               np.mean([x[10] for x in d_art[artista]]), # media/area convex
               np.mean([x[11] for x in d_art[artista]])] # media/razao convex

        D2.append(bar)
            
        i += 1

# gravamos D e D2 em pickle, serão nossas feature matrix!
f = open('dados/feature_matrix1.pkl', 'wb')
pk.dump(D, f)
f.close()

f = open('dados/feature_matrix2.pkl', 'wb')
pk.dump(D2, f)
f.close()

# #############################################
# # fazemos o pca apenas das entropias, apenas das distancias de picos e outras
# # medidas (D2), e de ambos
# d = np.array(D)
# comp, autoval, comp_prop, args = pc.pca_autoval(np.array(d))
# d2 = np.nan_to_num(D2)
# comp2, autoval2, comp2_prop, args2 = pc.pca_autoval(d2)
# d3 = np.concatenate((d, d2), axis=1)
# comp3, autoval3, comp3_prop, args3 = pc.pca_autoval(d3)

# # quantidade componentes
# print 'Qtd componentes (D, D2, D3):', len(comp_prop), len(comp2_prop), len(comp3_prop)

# # contribuição de cada componente
# print 'Var D:', [x*100 for x in np.around(comp_prop, decimals=2)], [x*100 for x in np.around(autoval, decimals=2)], args
# print 'Var D2:', [x*100 for x in np.around(comp2_prop, decimals=2)], [x*100 for x in np.around(autoval2, decimals=2)], args2
# print 'Var D3:', [x*100 for x in np.around(comp3_prop, decimals=2)], [x*100 for x in np.around(autoval3, decimals=2)], args3

# # GRAFICO PCA 1

# p.figure(figsize=(20,30))

# pintor = 0
# n = conf.NUM_PINTURAS
# # plot destacando cada pintor
# p.subplot(211)
# marca = 'o'
# for i in xrange(0,len(comp),n):
#     if i >= 120:
#         marca = 's'
#     p.plot(comp[:,0][i:i+n], comp[:,1][i:i+n], marca,
#            color=conf.cores[pintor], label=conf.artistas[pintor])
#     pintor += 1
# p.legend()

# # plot destacando cada movimento artístico
# p.subplot(212)
# # movimentos = ['Moderno', 'Barroco']
# # marcas = ['o', 's']
# # pintor = 0
# # for i in xrange(0,len(comp), len(comp)/2):
# #     print pintor
# #     p.plot(comp[:,0][i:i+n], comp[:,1][i:i+n], 'o',
# #            marker=marcas[pintor], label=movimentos[pintor])
# #     pintor += 1
# p.plot(comp[:,0][120:], comp[:,1][120:], 's', color='#000000', label='Barrocos')
# p.plot(comp[:,0][:120], comp[:,1][:120], 'o', color='#ffffff', label='Modernos')

# p.legend()
# p.savefig('pca1.png')

# # GRAFICO PCA 2
# p.clf()
# pintor = 0
# n = conf.NUM_PINTURAS
# marca = 'o'
# # plot destacando cada pintor
# p.subplot(211)
# for i in xrange(0,len(comp2),n):
#     if i >= 120:
#         marca = 's'
#     p.plot(comp2[:,0][i:i+n], comp2[:,1][i:i+n], marca,
#            color=conf.cores[pintor], label=conf.artistas[pintor])
#     pintor += 1
# p.legend()

# # plot destacando cada movimento artístico
# p.subplot(212)
# # movimentos = ['Moderno', 'Barroco']
# # marcas = ['o', 's']
# # pintor = 0
# # for i in xrange(0,len(comp), len(comp)/2):
# #     print pintor
# #     p.plot(comp[:,0][i:i+n], comp[:,1][i:i+n], 'o',
# #            marker=marcas[pintor], label=movimentos[pintor])
# #     pintor += 1
# p.plot(comp2[:,0][120:], comp2[:,1][120:], 's', color='#000000', label='Barrocos')
# p.plot(comp2[:,0][:120], comp2[:,1][:120], 'o', color='#ffffff', label='Modernos')
# p.legend()
# p.savefig('pca2.png')

# # GRAFICO PCA 3
# p.clf()
# pintor = 0
# marca = 'o'
# n = conf.NUM_PINTURAS
# # plot destacando cada pintor
# p.subplot(211)
# for i in xrange(0,len(comp3),n):
#     if i >= 120:
#         marca = 's'
#     p.plot(comp3[:,0][i:i+n], comp3[:,1][i:i+n], marca,
#            color=conf.cores[pintor], label=conf.artistas[pintor])
#     pintor += 1
# p.legend()

# # plot destacando cada movimento artístico
# p.subplot(212)
# # movimentos = ['Moderno', 'Barroco']
# # marcas = ['o', 's']
# # pintor = 0
# # for i in xrange(0,len(comp), len(comp)/2):
# #     print pintor
# #     p.plot(comp[:,0][i:i+n], comp[:,1][i:i+n], 'o',
# #            marker=marcas[pintor], label=movimentos[pintor])
# #     pintor += 1
# p.plot(comp3[:,0][120:], comp3[:,1][120:], 's', color='#000000', label='Barrocos')
# p.plot(comp3[:,0][:120], comp3[:,1][:120], 'o', color='#ffffff', label='Modernos')

# p.legend()
# p.savefig('pca3.png')


# # pca para bins

# d = np.array(dados)
# # criamos histogramas de 20 bins entre 0 e 20
# dh = [np.histogram(x, bins=20, range=(0,20))[0] for x in d]

# comp = pc.pca(dh)

# # pintor = 0
# # conf.cores = 'ygbc'

# # for i in xrange(0,len(dados),80):
# #     p.plot(comp[:,0][i:i+80], comp[:,1][i:i+80], 'o', color=conf.cores[pintor], label=conf.artistas[pintor])
# #     pintor += 1

# # p.legend()
# # p.show()

# ### segunda analise ###

# # pegando dados sobre os picos em si
# f = open('dados/picos2.pkl', 'rb')
# DD = pk.load(f)
# d2, n_segs = DD
# f.close()

# # nbr, C, L, V

# i = 0
# t = 0
# d_art = {}
# D = []
# # pintor
# for art in conf.artistas:
#     d_art[art] = []
#     # pinturas
#     for pint in range(conf.NUM_PINTURAS):
#         # segmentos (considerando todos os segmentos de todas as segmentacoes)
#         print n_segs[i]
#         for k in range(n_segs[i]):
#             nbr, C, L, V = d2[t]

#             xs = np.array([np.real(C[L[j]]) for j in range(nbr)])
#             ys = np.array([np.imag(C[L[j]]) for j in range(nbr)])
#             d_eucli = [np.sqrt((xs[j+1]-ys[j+1])**2 + (xs[j]-ys[j])**2)
#                        for j in range(nbr-1)]
#             d_pixel = [(L[j+1] - L[j])**2
#                        for j in range(nbr-1)]
#             d_eucli = np.array(d_eucli)
#             d_pixel = np.array(d_pixel)
            
#             foo = [np.mean(d_eucli), np.std(d_eucli),
#                    np.mean(d_pixel), np.std(d_pixel)]

#             d_art[art].append(foo)
#             D.append(foo)
#             t += 1
#         i += 1

# comp = pc.pca(np.nan_to_num(D))
# #p.clf()

# nump = conf.NUM_PINTURAS
# segs_pinturas = [np.sum(n_segs[i*nump:i*nump+nump]) for i in range(len(conf.artistas))]

# tudo = 0
# pintor = 0
# conf.cores = 'rbgc'
# for i in segs_pinturas:
#     print i, pintor, len(comp), tudo, tudo+i
#     p.plot(comp[:,0][tudo:tudo+i], comp[:,1][tudo:tudo+i], 'o', color=conf.cores[pintor], label=conf.artistas[pintor])
#     tudo += i
#     pintor += 1

# p.legend()
# p.show()
