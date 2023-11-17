import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
import astropy.cosmology.units as cu
from astropy.cosmology import FlatLambdaCDM

import sys
# sys.path.append('/content/drive/MyDrive/LE2023/dust/code')
sys.path.append(r"code\dust\code")
import var_constants as vc
import dust_constants as dc
import fix_constants as fc
import scattering_function as sf
import size_dist as sd
import calculate_scattering_function as csf

sys.path.append(r"code")
import surface_brightness as sb
# import brightness as fb
import geometry_nodelta as le_geo
import plot_nodelta as le_pl


alpha = 15
# # x = np.linspace(-10,10,1000)

# for ct in [120, 130, 140, 160, 180, 200, 230, 250, 280, 290, 320, 360, 440, 520]: #alpha = 0 done
for ct in [290, 320, 360, 440, 520]: #alpha = 0 done
# for al in [0, 15, 45, 75, 120, 310]:
    ctyi = ct * fc.dtoy
    a = np.tan(np.deg2rad(alpha))
    r_le2 = 2 * vc.z0ly * ctyi + (ctyi)**2 * (1 + a**2)
    r_le = np.sqrt(r_le2)
    print("starting %s"%(ct))
    xmin = -r_le-a*ctyi
    xmax = r_le-a*ctyi
    x = np.linspace(xmin,xmax,1000)
    new_xs_v838, new_ys_v838, surface_v838, act_v838, ange_v838, cossigma_v838 = le_geo.LE_xy_surface_concate_plane(alpha, vc.z0ly, ctyi, x)
    path_ge = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\num\\"
    np.save(path_ge+r"plane_nd_xs_v838_t%s.npy"%(ct), new_xs_v838)
    np.save(path_ge+r"plane_nd_ys_v838_t%s.npy"%(ct), new_ys_v838)
    np.save(path_ge+r"plane_nd_flux_v838_t%s.npy"%(ct),surface_v838)  
    np.save(path_ge+r"plane_nd_delta_v838_t%s.npy"%(ct), ange_v838)
    np.save(path_ge+r"plane_nd_cossigma_v838_t%s.npy"%(ct), cossigma_v838)
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    le_pl.plot(new_xs_v838, new_ys_v838, surface_v838, alpha, act_v838, axes, fig, save = False, name = "name")
    axes.set_title(r"V838 Mon - Plane: time$_{obs}$ = %s days"%(ct))
    # axes.set_title(r"V838 Mon - Plane: $\alpha$ = %s deg "%(al))
    name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"plane_nd_v838_t%s.pdf"%(ct)
    plt.savefig(name, dpi = 700)
    # plt.show()
print("end %s"%(ct))


for al in [0, 15, 45, 75, 120, 310]:
    a = np.tan(np.deg2rad(al))
    r_le2 = 2 * vc.z0ly * vc.ct + (vc.ct)**2 * (1 + a**2)
    r_le = np.sqrt(r_le2)
    print("starting %s"%(al))
    xmin = -r_le-a*vc.ct
    xmax = r_le-a*vc.ct
    x = np.linspace(xmin,xmax,1000)
    new_xs_v838, new_ys_v838, surface_v838, act_v838, ange_v838, cossigma_v838 = le_geo.LE_xy_surface_concate_plane(al, vc.z0ly, vc.ct, x)
    path_ge = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\num\\"
    np.save(path_ge+r"plane_nd_xs_v838_a%s.npy"%(al), new_xs_v838)
    np.save(path_ge+r"plane_nd_ys_v838_a%s.npy"%(al), new_ys_v838)
    np.save(path_ge+r"plane_nd_flux_v838_a%s.npy"%(al), surface_v838)
    np.save(path_ge+r"plane_nd_delta_v838_a%s.npy"%(al), ange_v838)
    np.save(path_ge+r"plane_nd_cossigma_v838_a%s.npy"%(al), cossigma_v838)
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    le_pl.plot(new_xs_v838, new_ys_v838, surface_v838, al, act_v838, axes, fig, save = False, name = "name")
    # axes.set_title(r"V838 Mon - Plane: time$_{obs}$ = %s days, $\alpha$ = %s deg,  z0 = %s pc"%(vc.Deltat, al, vc.z0))
    axes.set_title(r"V838 Mon - Plane: $\alpha$ = %s deg "%(al))
    name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"plane_nd_v838_a%s.pdf"%(al)
    plt.savefig(name, dpi = 700)
    # plt.show()
print("end %s"%(al))



# # x = np.linspace(-10,10,700)
# z = [-0.1, 0, 0.1, 1] # in pc
# zly = np.array(z) * fc.pctoly
# alpha = 15
# for zlyi in zly: #alpha = 0
#     p_name = round((zlyi / fc.pctoly),1)
#     print("starting %s"%(p_name))
#     new_xs_v838, new_ys_v838, surface_v838, act_v838, ange_v838, cossigma_v838 = le_geo.LE_xy_surface_concate_plane(alpha, zlyi, vc.ct, x)
#     if isinstance(surface_v838, int):
#         print("no LE")
#     else:
#         path_ge = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\num\\"
#         np.save(path_ge+r"plane_nd_xs_v838_z%s.npy"%(p_name), new_xs_v838)
#         np.save(path_ge+r"plane_nd_ys_v838_z%s.npy"%(p_name), new_ys_v838)
#         np.save(path_ge+r"plane_nd_surface_v838_z%s.npy"%(p_name), surface_v838)
#         np.save(path_ge+r"plane_nd_delta_v838_z%s.npy"%(p_name), ange_v838)
#         np.save(path_ge+r"plane_nd_cossigma_v838_z%s.npy"%(p_name), cossigma_v838)
#         fig, axes = plt.subplots(1, 1, figsize=(10, 8))
#         le_pl.plot(new_xs_v838, new_ys_v838, surface_v838, alpha, act_v838, axes, fig, save = False, name = "name")
#         axes.set_title(r"V838 Mon - Plane: time$_{obs}$ = %s days, $\alpha$ = %s deg,  z0 = %s pc"%(vc.Deltat, alpha, p_name))
#         name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"plane_nd_v838_z%s.pdf"%(p_name)
#         plt.savefig(name, dpi = 700)
#         plt.show()

# print("end %s"%(p_name))



# ## SPHERE
# # r = [0.1, 1, 3] # in pc
# # rly = np.array(r) * fc.pctoly
# for ct in [120, 200, 280, 360, 440, 520]: #alpha = 0 done
#     print("starting %s"%(ct))
#     cty = ct * fc.dtoy
#     # print(ct)
#     p_name = ct
#     print("starting %s"%(p_name))
#     new_xs_v838, new_ys_v838, surface_v838, ange_v838, cossigma_v838 = le_geo.LE_xy_surface_concate_sphere(vc.r0ly, cty, x)
#     path_ge = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\num\\"
#     np.save(path_ge+r"sphere_nd_xs_v838_t%s.npy"%(p_name), new_xs_v838)
#     np.save(path_ge+r"sphere_nd_ys_v838_t%s.npy"%(p_name), new_ys_v838)
#     np.save(path_ge+r"sphere_nd_surface_v838_t%s.npy"%(p_name), surface_v838)
#     np.save(path_ge+r"sphere_nd_delta_v838_t%s.npy"%(p_name), ange_v838)
#     np.save(path_ge+r"sphere_nd_cossigma_v838_t%s.npy"%(p_name), cossigma_v838)
#     fig, axes = plt.subplots(1, 1, figsize=(10, 8))
#     le_pl.plot_sphere(new_xs_v838, new_ys_v838, surface_v838, axes, fig, save = False, name = "name")
#     axes.set_title(r"V838 Mon - sphere: time$_{obs}$ = %s days"%(p_name))
#     name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"sphere_nd_v838_t%s.pdf"%(p_name)
#     plt.savefig(name, dpi = 700)
#     # plt.show()

# print("end %s"%(p_name))


