import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
import astropy.cosmology.units as cu
from astropy.cosmology import FlatLambdaCDM

import sys
# sys.path.append('/content/drive/MyDrive/LE2023/dust/code')
# sys.path.append(r"/Users/fbb/light_echo_modeling/fed/dust/code")
from setpath import path_to_LE
sys.path.append(path_to_LE + "/dust/code")


import var_constants as vc
import dust_constants as dc
import fix_constants as fc
import scattering_function as sf
import size_dist as sd
import calculate_scattering_function as csf

import surface_brightness as sb
# import brightness as fb
import geometry_nodelta as le_geo
import plot_nodelta as le_pl

alpha = 15
# # x = np.linspace(-10,10,1000)
ct = 290  # alpha = 0 done
    # for al in [0, 15, 45, 75, 120, 310]:
"""
ctyi = ct * fc.dtoy
a = np.tan(np.deg2rad(alpha))
r_le2 = 2 * vc.z0ly * ctyi + (ctyi) ** 2 * (1 + a ** 2)
r_le = np.sqrt(r_le2)

xmin = -r_le-a*ctyi
xmax = r_le-a*ctyi
x = np.linspace(xmin,xmax,100)
le_geo.LE_xy_surface_concate_plane(15, 1000, 1, x)

print("here")

sys.exit()
"""

# for ct in [120, 130, 140, 160, 180, 200, 230, 250, 280, 290, 320, 360, 440, 520]: #alpha = 0 done
for ct in [290]: #alpha = 0 done
# for al in [0, 15, 45, 75, 120, 310]:
    ctyi = ct * fc.dtoy
    a = np.tan(np.deg2rad(alpha))
    r_le2 = 2 * vc.z0ly * ctyi + (ctyi)**2 * (1 + a**2)
    r_le = np.sqrt(r_le2)
    print("starting ct=%s days, ctyi=%.2f years, radius %.2f (units?)"%(ct, ctyi, r_le))
    xmin = -r_le-a*ctyi
    xmax = r_le-a*ctyi
    x = np.linspace(xmin, xmax, 100)
    new_xs_v838, new_ys_v838, surface_v838, act_v838, ange_v838, cossigma_v838 = le_geo.LE_xy_surface_concate_plane_fed(alpha, vc.z0ly, ctyi, x)

    path_ge = r"%s/save_data/"%(path_to_LE)
    np.save(path_ge+r"/plane_nd_xs_v838_t%s.npy"%(ct), new_xs_v838)
    np.save(path_ge+r"/plane_nd_ys_v838_t%s.npy"%(ct), new_ys_v838)
    np.save(path_ge+r"/plane_nd_flux_v838_t%s.npy"%(ct),surface_v838)  
    np.save(path_ge+r"/plane_nd_delta_v838_t%s.npy"%(ct), ange_v838)
    np.save(path_ge+r"/plane_nd_cossigma_v838_t%s.npy"%(ct), cossigma_v838)
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    le_pl.plot(new_xs_v838, new_ys_v838, surface_v838, alpha, act_v838, axes, fig, save = False, name = "name")
    axes.set_title(r"V838 Mon - Plane: time$_{obs}$ = %s days"%(ct))
    # axes.set_title(r"V838 Mon - Plane: $\alpha$ = %s deg "%(al))
    name = path_to_LE+r"/figures/" + r"plane_nd_v838_t%s.pdf"%(ct)
    plt.savefig(name, dpi = 700)




    # plt.show()
print("end %s"%(ct))

sys.exit()
# for al in [0, 15, 45, 75, 120, 310]:
for al in [15]:
    a = np.tan(np.deg2rad(al))
    r_le2 = 2 * vc.z0ly * vc.ct + (vc.ct)**2 * (1 + a**2)
    r_le = np.sqrt(r_le2)
    print("starting %s"%(al))
    xmin = -r_le-a*vc.ct
    xmax = r_le-a*vc.ct
    x = np.linspace(xmin,xmax,1000)
    new_xs_v838, new_ys_v838, surface_v838, act_v838, ange_v838, cossigma_v838 = le_geo.LE_xy_surface_concate_plane(al, vc.z0ly, vc.ct, x)
    path_ge = path_to_LE+r"/save_data/"
    np.save(path_ge+r"plane_nd_xs_v838_a%s.npy"%(al), new_xs_v838)
    np.save(path_ge+r"plane_nd_ys_v838_a%s.npy"%(al), new_ys_v838)
    np.save(path_ge+r"plane_nd_flux_v838_a%s.npy"%(al), surface_v838)
    np.save(path_ge+r"plane_nd_delta_v838_a%s.npy"%(al), ange_v838)
    np.save(path_ge+r"plane_nd_cossigma_v838_a%s.npy"%(al), cossigma_v838)
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    le_pl.plot(new_xs_v838, new_ys_v838, surface_v838, al, act_v838, axes, fig, save = False, name = "name")
    # axes.set_title(r"V838 Mon - Plane: time$_{obs}$ = %s days, $\alpha$ = %s deg,  z0 = %s pc"%(vc.Deltat, al, vc.z0))
    axes.set_title(r"V838 Mon - Plane: $\alpha$ = %s deg "%(al))
    name = path_to_LE+r"/figures/" + r"plane_nd_v838_a%s.pdf"%(al)
    plt.savefig(name, dpi = 700)
    # plt.show()
print("end %s"%(al))

# print("end %s"%(p_name))


