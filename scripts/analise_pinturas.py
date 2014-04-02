# -*- coding: utf-8 -*-

from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage, fftpack
import pylab as p
import numpy as n
import imtools
import pickle as pk
from skimage.morphology import watershed, is_local_maximum, medial_axis, disk
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.filter import canny, threshold_otsu, rank
from scipy.misc import imsave
from sklearn.feature_extraction import image
from skimage.filter.rank import entropy
from skimage.transform import probabilistic_hough
#from sklearn.cluster import spectral_clustering
import matplotlib.pyplot as plt
import config
import mahotas.features as mf

fig = p.figure(num=None, figsize=(10, 10))
plot = fig.add_subplot(111)
plot.axis('off')

def entropia(viz):
    viz = list(viz)
    qtd = [viz.count(x) for x in viz]
    prob = [viz[i]/qtd[i] for i in range(len(viz))]
    return n.sum([-x*n.log(x) for x in prob if x != 0])

def covar(viz):
    return n.cov(viz, bias=1, rowvar=0)

def pearson(viz):
    shape = (3,3)
    matriz_viz = n.reshape(viz, shape)
    matriz_cov = n.cov(matriz_viz, bias=1, rowvar=0)
    stds = n.std(matriz_viz, 0)
    pear = n.zeros(shape)
    for i in xrange(shape[0]):
        for j in xrange(shape[0]):
            pear[i,j] = matriz_cov[i,j] / (stds[i] * stds[j])
    return pear.mean()

dados = []
n_segs = []
D2 = []
# abre a imagem, converte em tons de cinza e equaliza por histograma
for art in config.artistas:
    print art
    print '-'*10
    path = art.lower()
    mm = []

    for i in xrange(config.NUM_PINTURAS):
        print 'Pintura %s' % (i+1)
        im = ImageOps.equalize(Image.open('pinturas/%s/r.%s.jpg' % (path, i)).convert('L'))
        im_rgb = Image.open('pinturas/%s/r.%s.jpg' % (path, i))
        im_rgb_s = im_rgb.split()
        im_r = im_rgb_s[0]
        im_g = im_rgb_s[1]
        im_b = im_rgb_s[2]
        #im_r.save('r.jpg')
        #im_g.save('g.jpg')
        #im_b.save('b.jpg')
        #im_array = n.array(im)
        #im_array=((im_array-im_array.min())/(im_array.max()-im_array.min()))*.2
        #pearsons = ndimage.generic_filter(im, pearson, size=3)
        energias = fftpack.fft2(im).real**2 + fftpack.fft2(im).imag**2
        en_li = energias.sum(0)
        en_co = energias.sum(1)

        energias_r = fftpack.fft2(im_r).real**2 + fftpack.fft2(im_r).imag**2
        en_li_r = energias_r.sum(0)
        en_co_r = energias_r.sum(1)

        energias_g = fftpack.fft2(im_g).real**2 + fftpack.fft2(im_g).imag**2
        en_li_g = energias_g.sum(0)
        en_co_g = energias_g.sum(1)

        energias_b = fftpack.fft2(im_b).real**2 + fftpack.fft2(im_b).imag**2
        en_li_b = energias_b.sum(0)
        en_co_b = energias_b.sum(1)

        # mais uma... media das entropias locais

        ##### CARACTERISCAS GLOBAIS!! #######################################

        ### PREPROCESSAMENTO
        
        # converte para cinza
        im_cinza = im_rgb.convert('L')

        # equaliza
        im_eq = ImageOps.equalize(im_cinza)
            
        ### FILTRAGEM
        
        # filtro media
        im_media = Image.fromarray(ndimage.median_filter(im_eq, size=3))

        im_binaria = 255.*(n.array(im_media)>128)

        ### SEGMENTACAO / DETECCAO DE BORDAS
        
        # # segmentacao por slic
        # im_rgb_f = img_as_float(im_rgb)
        # labels = slic(im_rgb_f, convert2lab=True, ratio=25, n_segments=2, sigma=3.)
        # labels = labels + 1
        # # analisando os n_seg segmentos (regiao e contorno... per./area)
        # qr = []
        # seg = labels.copy()
        # conta_segs = 0
        # for k in xrange(1,5):
        #     seg = labels.copy()
        #     print k, n.sum(seg == k)
        #     seg[seg != k] = 0
        #     seg[seg == k] = 255
        #     imsave('pinturas/%s/seg%s.%s.png' % (path, k, i), seg)
            
        #     sx = ndimage.sobel(seg, axis=0, mode='constant')
        #     sy = ndimage.sobel(seg, axis=1, mode='constant')
        #     contorno = n.hypot(sx, sy)
        #     #imsave('cont.jpg', contorno)
            
        #     area = n.sum(seg != 0)
        #     peri = n.sum(contorno != 0)
        #     raz = float(peri**2)/area
        #     qr.append(raz)
        #     D2.append(area, peri)
        # # guardamos a media das razoes de perimetro/area de todos os segmentos
        # # da pintura
        # qrm = n.sum(qr) / k

        # entropia local (media)
        entropia_local = entropy(im_media, disk(5))
        sh = entropia_local.shape
        m_entropia_local = n.sum(entropia_local) / (sh[0]*sh[1])
        qel = m_entropia_local

        # entropia local (media) com disco maior, 50x50
        entropia_local_maior = entropy(im_media, disk(50))
        sh2 = entropia_local_maior.shape
        m_entropia_local_maior = n.sum(entropia_local_maior) / (sh2[0]*sh2[1])
        qel_maior = m_entropia_local_maior

        
        # usando canny/sobel + transformada de hough para detectar linhas de qualquer angulo
        #edges = canny(im_binaria, 2, 1, 25)
        #edges = canny(im_binaria, 0.3, 0.2)
        sx = ndimage.sobel(im_binaria, axis=0, mode='constant')
        sy = ndimage.sobel(im_binaria, axis=1, mode='constant')
        edges = n.hypot(sx, sy)
                
        lines = probabilistic_hough(edges, threshold=10, line_length=80, line_gap=3)
        #plt.figure(figsize=(10, 10))
        #plt.imshow(edges * 0)
        #plot.clear()
        #plot.axis('off')
        #for line in lines:
        #    p0, p1 = line
        #    plot.plot((p0[0], p1[0]), (p0[1], p1[1]))
        #fig.savefig('pinturas/%s/hough.%s.png' % (path, i))
        #print area, peri, raz, len(lines)
        ql = len(lines)

        # calculando agora os haralick
        h = mf.haralick(n.array(im_media))

        m = [#pearsons.mean(), # media pearson
             #pearsons.std(), # std pearson
            # CINZA
            imtools.entropy(im), # entropia
            en_li.mean(), # media energias das linhas
            en_li.std(), # std energias das linhas
            en_co.mean(), # media energias das colunas
            en_co.std(), # std energias das colunas
            # centroide linha
            n.sum([en_li[j]*j for j in range(len(en_li))]) / n.sum(en_li),
            # centroide coluna
            n.sum([en_co[j]*j for j in range(len(en_co))]) / n.sum(en_co),
            # media energias total
            energias.mean(),
            # std energias total
            energias.std(),
            # RED
            imtools.entropy(im_r), # entropia
            en_li_r.mean(), # media energias das linhas
            en_li_r.std(), # std energias das linhas
            en_co_r.mean(), # media energias das colunas
            en_co_r.std(), # std energias das colunas
            # centroide linha
            n.sum([en_li_r[j]*j for j in range(len(en_li_r))]) / n.sum(en_li_r),
            # centroide coluna
            n.sum([en_co_r[j]*j for j in range(len(en_co_r))]) / n.sum(en_co_r),
            # media energias total
            energias_r.mean(),
            # std energias total
            energias_r.std(),
            # GREEN
            imtools.entropy(im_g), # entropia
            en_li_g.mean(), # media energias das linhas
            en_li_g.std(), # std energias das linhas
            en_co_g.mean(), # media energias das colunas
            en_co_g.std(), # std energias das colunas
            # centroide linha
            n.sum([en_li_g[j]*j for j in range(len(en_li_g))]) / n.sum(en_li_g),
            # centroide coluna
            n.sum([en_co_g[j]*j for j in range(len(en_co_g))]) / n.sum(en_co_g),
            # media energias total
            energias_g.mean(),
            # std energias total
            energias_g.std(),
            # BLUE
            imtools.entropy(im_b), # entropia
            en_li_b.mean(), # media energias das linhas
            en_li_b.std(), # std energias das linhas
            en_co_b.mean(), # media energias das colunas
            en_co_b.std(), # std energias das colunas
            # centroide linha
            n.sum([en_li_b[j]*j for j in range(len(en_li_b))]) / n.sum(en_li_b),
            # centroide coluna
            n.sum([en_co_b[j]*j for j in range(len(en_co_b))]) / n.sum(en_co_b),
            # media energias total
            energias_b.mean(),
            # std energias total
            energias_b.std(),
            # MEDIDAS GLOBAIS
            #qrm, # media das razoes perimetro/area dos K segmentos da cada obra
            qel, # media das entropias locais (disco 5x5)
            qel_maior, # media das entropias locais (disco 50x50)
            ql # quantidade de linhas (segundo transf. de Hough) de cada obra
            ] + list(h.flatten()) # e também agora... haralick
            
        print m
        mm.append(m)

    features = [art, mm]
    dados.append(features)
    
print dados

f = open('dados/pinturas.pkl', 'wb')
pk.dump(dados, f)
f.close()

# mostra imagem filtrada
# p.figure()
# p.imshow(medias)
# p.gray()
# p.axis('equal')
# p.axis('off')
# p.show()

#medias = ndimage.median_filter(im, size=5)
#gaussianas = ndimage.gaussian_filter(im, sigma=1.)
#entropias = ndimage.generic_filter(im, entropia, size=3)
