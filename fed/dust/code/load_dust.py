import numpy as np

import sys
from setpath import path_to_LE
sys.path.append(path_to_LE + "/dust/code")

#path_to_LE = "/Users/fbb/light_echo_modeling/fed"
#sys.path.append(path_to_LE + r"/dust/code")

path_dustdata = path_to_LE + r"/dust/data"

# older models used for g (degree of forward scattering) values
# carbonaceous dust
carbonQ = path_dustdata + r'/dustmodels_WD01/Gra_81.dat'

# silicate dust
siliconQ = path_dustdata + r'/dustmodels_WD01/suvSil_81.dat'

# older models used for g (degree of forward scattering) values
# carbonaceous dust
carbong = path_dustdata + r'/dustmodels_WD01/Gra_81.dat'
# silicate dust
silicong = path_dustdata + r'/dustmodels_WD01/suvSil_81.dat'

# pull out available wavelengths for g values, convert to cm from um, and take the
waveg = np.loadtxt(path_dustdata + r'/dustmodels_WD01/LD93_wave.dat', unpack=True)  # micronm
# pull out available sizes for the g values, convert to cm from um, and take the log
sizeg = np.loadtxt(path_dustdata + r'/dustmodels_WD01/LD93_aeff.dat', unpack=True)  # micron

Qcarb_sca = np.loadtxt(carbonQ, usecols=(2), unpack=True)
Qcarb_abs = np.loadtxt(carbonQ, usecols=(1), unpack=True)
Qsil = np.loadtxt(siliconQ, usecols=(2), unpack=True)

gcarb = np.loadtxt(carbong, usecols=(3), unpack=True)
gsil = np.loadtxt(silicong, usecols=(3), unpack=True)


