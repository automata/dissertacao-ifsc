# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import scipy.stats as ss
import matplotlib.mlab as mlab
import string
import config

pintores = [string.lower(config.artistas[i]) for i in config.ordem][6:]
for pin in pintores:
    ns = []
    for i in range(20):
        # im = ImageOps.equalize(Image.open('pinturas/caravaggio/r.%s.jpg' % i).convert('L'))
        im = Image.open('pinturas/%s/r.%s.jpg' % (pin, i)).convert('L')
        # im = mpimg.imread('pinturas/gogh/r.%s.jpg' % i)
        im = np.asarray(im)

        x = im.flatten()
        n, bins = np.histogram(x, 256, normed=True)
        ns.append(n)
        #n, bins, patches = plt.hist(x, 256, normed=1, histtype='stepfilled', alpha=0.4, facecolor=cores[j], label="Caravaggio")
        
        # y = mlab.normpdf(bins, x.mean(), x.std())
        # plt.plot(bins, y, 'r--')
    foo = np.array(ns).mean(0)
    if pin == 'gogh':
        plt.plot(bins[:256], foo[:256], label='Van Gogh')
    elif pin == 'miro':
        plt.plot(bins[:256], foo[:256], label=u'Miró')
    else:
        plt.plot(bins[:256], foo[:256], label=string.capitalize(pin))
plt.legend()

plt.xlim((0,256))
plt.legend()
plt.show()
    
        
