import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from astropy import units as u
from scipy.special import erf
from scipy import integrate

import sys
sys.path.append(r"C:\\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\code\\dust\\code")

import var_constants as vc
import dust_constants as dc
import fix_constants as fc
import scattering_function as sf
import size_dist as sd


def calculate_scattering_function(mu, sizeg, waveg, wave, Qcarb, gcarb):
    B1 = sd.Bi_carb(dc.a01, dc.bc1)
    # calculate B2
    B2 = sd.Bi_carb(dc.a02, dc.bc2)
    carbon_distribution = [sd.Dist_carb(idx, B1, B2).value for idx in 1e-4*sizeg*u.cm] #in cm

    ds = sf.dS(mu, sizeg, waveg, wave, Qcarb, gcarb, carbon_distribution)

    S = sf.S(ds, sizeg)

    return ds, S


def main(mu, wave):
    # path_dustdata = '/content/drive/MyDrive/LE2023/dust/data/'
    path_dustdata = r"C:\\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\code\\dust\\data"
    # pull out available wavelengths for g values, convert to cm from um, and take the 
    waveg = np.loadtxt(path_dustdata+r'\\dustmodels_WD01\\LD93_wave.dat', unpack=True) #micronm
    # pull out available sizes for the g values, convert to cm from um, and take the log
    sizeg = np.loadtxt(path_dustdata+r'\\dustmodels_WD01\\LD93_aeff.dat', unpack=True) #micron

    # older models used for g (degree of forward scattering) values
    # carbonaceous dust
    carbonQ = path_dustdata+r'\\dustmodels_WD01\\Gra_81.dat'
    Qcarb_sca = np.loadtxt(carbonQ, usecols=(2), unpack=True)
    Qcarb_abs = np.loadtxt(carbonQ, usecols=(1), unpack=True)
    Qcarb = Qcarb_sca / (Qcarb_sca + Qcarb_abs)

    # silicate dust
    siliconQ = path_dustdata+r'\\dustmodels_WD01\\suvSil_81.dat'
    Qsil = np.loadtxt(siliconQ, usecols=(2), unpack=True)

    # older models used for g (degree of forward scattering) values
    # carbonaceous dust
    carbong = path_dustdata+r'\\dustmodels_WD01\\Gra_81.dat'
    gcarb = np.loadtxt(carbong, usecols=(3), unpack=True)
    # silicate dust
    silicong = path_dustdata+r'\\dustmodels_WD01\\suvSil_81.dat'
    gsil = np.loadtxt(silicong, usecols=(3), unpack=True)

    ds, S = calculate_scattering_function(mu, sizeg, waveg, wave, Qcarb, gcarb)

    return ds, S


# if __name__ == "__main__":
#     ds, S = main(mu, wave)
