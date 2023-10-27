import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
import astropy.cosmology.units as cu
from astropy.cosmology import FlatLambdaCDM

import sys
# sys.path.append('/content/drive/MyDrive/LE2023/dust/code')
sys.path.append(r"C:\\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\code\\dust\\code")
import var_constants as vc
import dust_constants as dc
import fix_constants as fc
import scattering_function as sf
import size_dist as sd
import calculate_scattering_function as csf

sys.path.append(r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\code")
import surface_brightness as sb
import geometry as le_geo
import plot as le_pl



deltass = np.linspace(0, 90, 200)
x = np.linspace(-6,6,600)

r = [0.1, 1, 3] # in pc
rly = np.array(r) * fc.pctoly
for rlyi in rly: #alpha = 0
    print(rlyi)
    p_name = round((rlyi / fc.pctoly),1)
    print("starting %s"%(p_name))
    new_xs_v838, new_ys_v838, surface_v838, fin_delta_v838, cossigma_v838 = le_geo.LE_xy_surface_concate_sphere(rlyi, vc.ct, deltass, x)
    path_ge = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\num\\"
    np.save(path_ge+r"sphere_xs_v838_r%s.npy"%(p_name), new_xs_v838)
    np.save(path_ge+r"sphere_ys_v838_r%s.npy"%(p_name), new_ys_v838)
    np.save(path_ge+r"sphere_surface_v838_r%s.npy"%(p_name), surface_v838)
    np.save(path_ge+r"sphere_delta_v838_r%s.npy"%(p_name), fin_delta_v838)
    np.save(path_ge+r"sphere_cossigma_v838_r%s.npy"%(p_name), cossigma_v838)
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    le_pl.plot_sphere(new_xs_v838, new_ys_v838, surface_v838, deltass, axes, fig, save = False, name = "name")
    axes.set_title(r"V838 Mon - sphere: time$_{obs}$ = %s days, r0 = %s pc"%(vc.Deltat, p_name))
    name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"sphere_v838_r%s.pdf"%(p_name)
    plt.savefig(name, dpi = 700, bbox_inches='tight')
    plt.show()

print("end %s"%(p_name))




