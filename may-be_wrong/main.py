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



deltass = np.linspace(65, 70, 200)
# x = np.linspace(-3,3,600)

for al in [0, 15, 45, 75, 120, 310]: #alpha = 0 done
    print("starting %s"%(al))
    a = np.tan(np.deg2rad(al))
    r_le2 = 2 * vc.z0ly * vc.ct + (vc.ct)**2 * (1 + a**2)
    r_le = np.sqrt(r_le2)
    # print("starting %s"%(al))
    xmin = -r_le-a*vc.ct
    xmax = r_le-a*vc.ct
    x = np.linspace(xmin,xmax,800)
    new_xs_v838, new_ys_v838, flux_v838, act_v838, fin_delta_v838, cossigma_v838, ange_v838 = le_geo.LE_xy_surface_concate_plane(al, vc.z0ly, vc.ct, deltass, x)
    path_ge = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\num\\"
    np.save(path_ge+r"plane_xs_v838_delta_a%s.npy"%(al), new_xs_v838)
    np.save(path_ge+r"plane_ys_v838_delta_a%s.npy"%(al), new_ys_v838)
    np.save(path_ge+r"plane_flux_v838_delta_a%s.npy"%(al), flux_v838)
    np.save(path_ge+r"plane_delta_v838_delta_a%s.npy"%(al), fin_delta_v838)
    np.save(path_ge+r"plane_cossigma_v838_delta_a%s.npy"%(al), cossigma_v838)
    np.save(path_ge+r"plane_ange_v838_delta_a%s.npy"%(al), ange_v838)

    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    le_pl.plot(new_xs_v838, new_ys_v838, flux_v838, deltass, al, act_v838, axes, fig, save = False, name = "name")
    axes.set_title(r"V838 Mon - Plane: time$_{obs}$ = %s days, delta = %s, $\alpha$ = %s deg, z0 = %s pc"%(vc.Deltat, [deltass.min(), deltass.max()],al, vc.z0))
    name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"plane_v838_delta_a%s.pdf"%(al)
    # plt.savefig(name, dpi = 700, bbox_inches='tight')
    plt.show()
print("end %s"%(al))



# z = [-0.1, 0, 0.1, 1] # in pc
# zly = np.array(z) * fc.pctoly
# alpha = 15
# for zlyi in zly: #alpha = 0
#     p_name = round((zlyi[0] / fc.pctoly),1)
#     print("starting %s"%(p_name))
#     new_xs_v838, new_ys_v838, surface_v838, act_v838, fin_delta_v838, cossigma_v838 = le_geo.LE_xy_surface_concate_plane(alpha, zlyi[0], vc.ct, deltass, x)
#     path_ge = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\num\\"
#     np.save(path_ge+r"plane_xs_v838_z%s.npy"%(p_name), new_xs_v838)
#     np.save(path_ge+r"plane_ys_v838_z%s.npy"%(p_name), new_ys_v838)
#     np.save(path_ge+r"plane_surface_v838_z%s.npy"%(p_name), surface_v838)
#     np.save(path_ge+r"plane_delta_v838_z%s.npy"%(p_name), fin_delta_v838)
#     np.save(path_ge+r"plane_cossigma_v838_z%s.npy"%(p_name), cossigma_v838)
#     fig, axes = plt.subplots(1, 1, figsize=(10, 8))
#     le_pl.plot(new_xs_v838, new_ys_v838, surface_v838, deltass, alpha, act_v838, axes, fig, save = False, name = "name")
#     axes.set_title(r"V838 Mon - Plane: time$_{obs}$ = %s days, $\alpha$ = %s deg,  z0 = %s pc"%(vc.Deltat, alpha, p_name))
#     name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"plane_v838_z%s.pdf"%(p_name)
#     plt.savefig(name, dpi = 700, bbox_inches='tight')
#     plt.show()
# print("end %s"%(p_name))




