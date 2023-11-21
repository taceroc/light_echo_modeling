import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from load_dust import *
from astropy import units as u
from scipy.special import erf
from scipy import integrate

import sys
from setpath import path_to_LE
sys.path.append(path_to_LE + "/dust/code")

#path_to_LE = "/Users/fbb/light_echo_modeling/fed"
#sys.path.append(path_to_LE + r"/dust/code")

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

#path_to_LE = "path_to_LE"
def main(mu, wave):
    Qcarb = Qcarb_sca / (Qcarb_sca + Qcarb_abs)

    ds, S = calculate_scattering_function(mu, sizeg, waveg, wave, Qcarb, gcarb)

    return ds, S


# if __name__ == "__main__":
#     ds, S = main(mu, wave)
