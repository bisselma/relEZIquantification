import numpy as np
from pylab import imshow
import mahotas as mh

yy, xx = np.meshgrid(np.arange(0,10), np.arange(0,10))

circs =  ((yy - 4) ** 2) + ((xx - 4)**2) <= 3**2

dist = ((yy - 4) ** 2) + ((xx - 4)**2)**0.5
dmap = mh.distance(circs)

breakpoint