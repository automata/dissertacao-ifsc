# encoding:utf-8
# curvatura!!

import copy
from scipy import ndimage
import numpy as np
import Image
import matplotlib.pyplot as plt
import config
import pickle as pk
import mahotas.polygon as mp

np.set_printoptions(threshold=np.nan)

fig = plt.figure(num=None, figsize=(10, 10))
fig2 = plt.figure(figsize=(10,10))
#plot = fig.add_subplot(111)

D = []
D2 = [] # usada para armazenar todos os dados do calculo de curvaturas
n_segs = [] # usada para saber qual número de segmentos por pintura

def filter_by_size(im_label, size):
    new = np.zeros(im_label.shape)
    lab = np.unique(im_label)
    for k in lab:
        x, y = np.where(im_label == k)
        if len(x) < size:
            new[x, y] = 0
        else:
            new[x, y] = k
    return new


def maximos(List, limiar):
    lista_picos = []
    valor_picos = []
    x = len(List)
    for i in range(1,(x - 2)):
        if abs(List[i]) > abs(List[i - 1]) and abs(List[i]) > abs(List[i + 1]) and abs(List[i]) > limiar:
            lista_picos.append(i)
            valor_picos.append(List[i])

    if abs(List[0]) > abs(List[1]) and abs(List[0]) > abs(List[x - 1]) and abs(List[0]) > limiar:
        lista_picos.append(0)
        valor_picos.append(List[0])

    if abs(List[x - 1]) > abs(List[0]) and abs(List[x - 1]) > abs(List[x - 2]) and abs(List[x - 1]) > limiar:
        lista_picos.append(x - 1)
        valor_picos.append(List[x - 1])

    nbr = len(lista_picos)

    return nbr, lista_picos, valor_picos


def chainpoint(p, d):
    d = d % 8

    x = p.real
    y = p.imag

    MATRIZCOMPLEXA = np.array([[y], [y - 1], [y - 1], [y - 1], [y], [y + 1], [y + 1], [y + 1]])
    MATRIZREAL = np.array([[x + 1], [x + 1], [x], [x - 1], [x - 1], [x - 1], [x], [x + 1]])

    pnext = complex(MATRIZREAL[d], MATRIZCOMPLEXA[d])
    return pnext


def find_next(pc, dpc, imagem, label, back):
    dcp = ((dpc + 4) % 8)
    for r in range(7):
        de = ((dcp + r) % 8)
        dint = ((dcp + r + 1) % 8)
        pe = chainpoint(pc, de)
        pint = chainpoint(pc, dint)
        if imagem[pint.real, pint.imag] == label and imagem[pe.real, pe.imag] == back: #imagem E = x0 + y0j
            pn = pe
            dcn = de
    return pn, dcn


def contorno(imagem, label, back):  # entre imagem binaria 0 preto, 255 branco
    E = []
    # imagem = np.array(imagem)
    linha, coluna = imagem.shape
    v = 0
    for i in range(linha - 1):
        if v == 2:
            break
        for j in range(coluna - 1):
            if imagem[i, j] == back and imagem[i, j + 1] == label:
                x0 = i
                y0 = j
                v = v + 1
                break

    E.append(complex(x0, y0))
    n = 1
    dcn = 4
    
    # encontra candidato a borda
    verificador = 0
    while verificador == 0:
        cand = chainpoint(E[0], dcn)  # cand borda
        dnc = dcn + 1
        if dnc == 8:
            dnc = 0
        objeto = chainpoint(E[0], dnc)  # cand objeto
        if imagem[int(objeto.real), int(objeto.imag)] == label and imagem[int(cand.real), int(cand.imag)] == back:
            verificador = 1
        else:
            dcn = dcn + 1

    next_pixel = copy.copy(cand)

    while (next_pixel != E[0]):
        E.append(next_pixel)
        dpc = copy.copy(dcn)
        next_pixel, dcn = find_next(E[n], dpc, imagem, label, back)
        n = n + 1
    return E


def perimetro(vetor):
    p = 0
    for i in range(1, len(vetor)):
        p = p + abs(vetor[i] - vetor[i - 1])
    #sum([abs(vetor[i - vetor[i - 1]) for i in range(1,len(vetor))])

    return p


def unshift(vetor):
    L = int(np.max(len(vetor)))
    metade = int(np.floor(L / 2))
    b = [None] * len(vetor)

    for i in range(0, (L - metade)):
        b[i] = vetor[i + metade]

    for i in range(0, metade):
        b[i + (L - metade)] = vetor[i]

    return b


def curvatura(vetor, sigma):
    Npx = len(vetor)
    vetor = np.array(vetor)
    U = np.fft.fft(vetor)
    U0 = np.fft.fftshift(U)

    s = range(int(-np.floor(Npx / 2)), int((Npx - np.floor(Npx / 2))))

    aux = np.array([(2j * np.pi) * s[i] for i in range(len(s))])
    dU = aux * U0

    aux2 = np.array([((2j * np.pi) * s[i]) ** 2 for i in range(len(s))])
    ddU = aux2 * U0

    aux3 = np.array([x ** 2 for x in s])
    G = np.exp([-(2 * np.pi) ** 2 * aux3[i] / (2 * sigma ** 2) for i in range(len(s))])

    Ufiltro = U0 * G
    dUfiltro = dU * G
    ddUfiltro = ddU * G

    uf = np.fft.ifft(unshift(Ufiltro))

    deri1 = np.fft.ifft(unshift(dUfiltro))
    deri2 = np.fft.ifft(unshift(ddUfiltro))

    C = perimetro(vetor) / perimetro(uf)

    deri1 = deri1 * C
    deri2 = deri2 * C

    aux4 = -np.imag(deri1 * np.conj(deri2))
    aux5 = [x ** 3 for x in np.abs(deri1)]
    k = aux4 / aux5

    return k

###############################################################################
###############################################################################

for art in config.artistas:
    D2 = [] # usada para armazenar todos os dados do calculo de curvaturas
    n_segs = [] # usada para saber qual número de segmentos por pintura

    print '\n\n############################## Pintor %s' % art
    path = art.lower()
    # quantas pinturas de cada pintor?
    for pint in range(config.NUM_PINTURAS):
        print '`- Quadro %s' % pint
        # quais grupos de segmentos? (neles existem as regioes conexas)
        conta_segs = 0
        for k in range(1,5):
            print ' `- Segmento %s' % k
            valid = Image.open('pinturas/%s/seg%s.%s.png' % (path, k, pint)).convert('L')
        
            # image 0 fundo, 255 obj, intermediarios --> 1 obj e 0 fundo intermediarios fundo
            valid = valid.point(lambda p: p > 1 and 1).convert("L")  # threshold
            valid = np.array(valid)  # transform as array
            
            
            # preenche linha preta nas bordas
            linha, coluna = valid.shape
            valid[:, (coluna - 1)] = 0
            valid[:, 0] = 0
            valid[:, (coluna - 2)] = 0
            valid[:, 1] = 0
            
            valid[(linha - 1), :] = 0
            valid[0, :] = 0
            valid[(linha - 2), :] = 0
            valid[1, :] = 0
                    
            # filled holes
            # filled = sp.ndimage.morphology.binary_fill_holes(valid)
            filled = ndimage.morphology.binary_fill_holes(valid, structure=np.ones((3,3))).astype(int)
            
            # find  and mark labels
            objects, num_objects = ndimage.label(valid)
            
            # filter By size
            size = 250
            filt = filter_by_size(objects, size)
            
            # determina os labels da imagem filtrada
            labels = np.unique(filt)
            labels = np.uint32(labels)
            
            # objects_slices = sp.ndimage.find_objects(filt)
            
            # for obj_slices in objects_slices:
            #     print thre[obj_slices]

            
            plot = fig.add_subplot(2, 2, 3)
            plot.imshow(objects)
            plot.set_title("Objetos Localizados")
            
            plot = fig.add_subplot(2, 2, 4)
            plot.imshow(filt)
            plot.set_title("Objetos filtrados size = %s" % size)
            
            plot = fig.add_subplot(2, 2, 2)
            plot.imshow(valid)
            plot.set_title("Burados preenchidos")
            
            plot = fig.add_subplot(2, 2, 1)
            plot.imshow(valid)
            plot.set_title("Threshold Manual")
            # nome do pintor . operacao . num da pintura . num do grupo de segs
            fig.savefig("saidas/curvaturas/%s.tratamento.%s.%s.png" % (path, pint, k))

            # print np.unique(objects)
            # print labels
            
            qtd_picos = []

            # exclui o primeiro elemento indice 0 (background)
            labels = np.delete(labels, [0])
            print 'labels', len(labels)
            for i in labels:
                print 'foo', i
                # contorno, curvatura
                C = contorno(filt, i, 0)
                
                Kurv = curvatura(C, 50)
                
                nbr, L, V = maximos(Kurv, 0.02)

                # area, perimetro, convex hull
                sss = filt.copy()
                sss[filt != i] = 0
                sss[filt == i] = 1
                sss = sss.astype(int)

                # por enquanto removendo convex hull, pois está bugado qhull
                #convex = convex_hull_image(sss)
                convex = mp.fill_convexhull(sss)
                print 'convex hull feito'
                Aor = np.sum(sss)
                Ac = np.sum(convex)
                
                # guarda a quantidade de picos
                qtd_picos.append(nbr)
                # fig2.clf()
                # plot2 = fig2.add_subplot(221)
                # plot2.imshow(convex, cmap=plt.cm.gray, interpolation='nearest')
                # plot2 = fig2.add_subplot(222)
                # plot2.imshow(sss, cmap=plt.cm.gray, interpolation='nearest')
                # plot2 = fig2.add_subplot(223)
                # fig2.savefig('foo.png')
                # perímetro
                peri = len(C)
                D2.append([nbr, C, peri, L, V, Aor, Ac, float(Aor) / Ac, np.sqrt(peri**2/Aor)])
                #D2.append([nbr, C, peri, L, V, Aor, np.sqrt(peri**2/Aor)])
                
                conta_segs += 1
                #plt.figure(i)
                #plt.clf()
                
                x = np.real(C)
                y = np.imag(C)

                fig.clf()
                
                plot = fig.add_subplot(1, 3, 1)
                plot.plot(x, y)
                for j in range(nbr):
                    x1 = np.real(C[L[j]])
                    y1 = np.imag(C[L[j]])
                    plot.plot(x1, y1, 'xr', linewidth=5)
                plot.set_title("Regiao")
                
                plot = fig.add_subplot(1, 3, 2)
                plot.plot(Kurv)
                plot.plot(L, V, 'xr', linewidth=5)
                plot.set_title("Curvatura (picos)")
                
                plot = fig.add_subplot(1, 3, 3)
                plot.hist(Kurv)
                plot.set_title("Curvatura")
                
                # nome do pintor . op . num pintura . num grupo segs . num seg
                fig.savefig("saidas/curvaturas/%s.curvatura.%s.%s.%s.png" % (path, pint, k, i))
                
            # histograma dos picos
            plt.figure(0)
            plt.clf()
            plt.hist(qtd_picos)
            # pintor . op . num pintura . num grupo segs
            plt.savefig("saidas/curvaturas/%s.histpicos.%s.%s.png" % (path, pint, k))
            print 'qtd picos', qtd_picos
            D.append(qtd_picos)
        n_segs.append(conta_segs)
    DD = (D2, n_segs)
    f = open('dados/segs_%s.pkl' % path, 'wb')
    pk.dump(DD, f)
    f.close()


#f = open('dados/picos.pkl', 'wb')
#pk.dump(D, f)
#f.close()

# DD = (D2, n_segs)
# f = open('dados/segs.pkl', 'wb')
# pk.dump(DD, f)
# f.close()
