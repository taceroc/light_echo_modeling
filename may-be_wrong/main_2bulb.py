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


def calc_intersection_xz_plane(x_p, y_p, r0ly, ct, location_bulb):
    """
    Calculate the intersection points x,y,z between the plane and the paraboloid

    Arguments:
        x, y: initialize values for x, e.g: x = np.linspace(-10, 10, 1000) in ly
        r0: radii sphere in ly
        ct: time where the LE is observed
        location_bulb: index of the location in the ct paraboloid of the sphere's center

    Return:
        x_inter, y_inter, z_inter: intersection plane and sphere
        angl: angle between line of sight and source-dust
    """
    # location [43,70]
    x_p, y_p = np.meshgrid(x_p, y_p)
    z_p = (x_p**2 + y_p**2 - ct**2) / (2 * ct)
    z_e = np.sqrt(r0ly**2 - (x_p-x_p[location_bulb[0], location_bulb[1]])**2 - (y_p-y_p[location_bulb[0], location_bulb[1]])**2) + z_p[location_bulb[0], location_bulb[1]]
    z_e2 = -np.sqrt(r0ly**2 - (x_p-x_p[location_bulb[0], location_bulb[1]])**2 - (y_p-y_p[location_bulb[0], location_bulb[1]])**2) + z_p[location_bulb[0], location_bulb[1]]
    # Define the radius and center of the sphere
    h = x_p[location_bulb[0], location_bulb[1]]
    k = y_p[location_bulb[0], location_bulb[1]]
    l = z_p[location_bulb[0], location_bulb[1]]
    print(h,k,l)

    center = (h, k, l)

    # Find points of intersection by scanning all the points and extract the ones that are inside the sphere and the paraboloid
    intersection_points = []

    for i in range(len(x_p)):
        for j in range(len(y_p)):
            x_par, y_par, z_par = x_p[i,j], y_p[i,j], z_p[i,j]

            # Check if the point is inside both the sphere and the paraboloid
            sphere_condition = ((x_par - center[0])**2 + (y_par - center[1])**2 + (z_par - center[2])**2) <= r0ly**2
            paraboloid_condition = (x_par**2 + y_par**2) <= (ct**2 + 2 * ct * z_par)

            if (sphere_condition and paraboloid_condition):
                intersection_points.append((x_par, y_par, z_par))

    x_inter = np.array([inter[0] for inter in intersection_points])
    y_inter = np.array([inter[1] for inter in intersection_points])
    z_inter = np.array([inter[2] for inter in intersection_points])


    # Create a contour plot in the X-Y plane (Z=0)
    plt.figure()
    plt.contourf(x_p, y_p, z_e, levels = 10, cmap='magma', alpha=0.5)
    plt.contourf(x_p, y_p, z_e2, levels = 10, cmap='magma', alpha=0.5)

    plt.colorbar(label='Z1')
    # plt.contour(x_p, y_p, z_p, levels = 10, colors='red', linewidths=2, label='Intersection')
    plt.scatter(x_inter, y_inter, alpha = 0.1)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Intersection center at x0 = %.2f, y0 = %.2f, z0 = %.2f'%(h,k,l))
    plt.grid(True)
    plt.axis('equal')

    name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"sphere_ncenter_countour_r%s.pdf"%([r0ly, location_bulb])
    # Show the plot
    # plt.savefig(name, dpi = 700)
    # plt.show()

    # calculate the angle between the line of sight (z) and the vector source-dust (x,y,z)
    angl = np.arccos(z_inter / np.sqrt(x_inter**2 + y_inter**2 + z_inter**2))

    return x_inter, y_inter, z_inter, angl, x_p, y_p, z_e, z_e2


def rinout_blub(x_inter, y_inter, z_inter):
    """
    Calculate the x,y projections

    Arguments:
        x_inter, y_inter, z_inter: intersection paraboloid + dust in ly
    
    Return:
        new_xs, new_ys: projections
    """
    # import this fancy thing to convert ly to arcsec, almost the same as using  np.arctan(r_le_out / vc.d) * (180 / np.pi) * 3600
    cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)
    d = (vc.d * u.lyr).to(u.Mpc)
    reds = d.to(cu.redshift, cu.redshift_distance(cosmo, kind="comoving"))
    # linear size = angular_size * d_A
    d_A = cosmo.angular_diameter_distance(z=reds)

    phis = np.arctan2(y_inter, x_inter)

    new_xs = (x_inter * u.lyr).to(u.Mpc)
    # distance_Mpc = d_A * theta_radian
    new_xs = (new_xs / d_A ).value / (np.pi / 180 / 3600)

    new_ys = (y_inter * u.lyr).to(u.Mpc)
    new_ys = (new_ys / d_A ).value / (np.pi / 180 / 3600)

    return phis, new_xs, new_ys


def plot(surface, new_xs, new_ys, ax, fig, save = False, name = "name"):
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    surface_300_norm = ( surface.copy() - np.nanmin(surface.copy())  ) / (np.nanmax(surface.copy()) - np.nanmin(surface.copy()))
    # cmap = matplotlib.colormaps.get_cmap('magma_r')
    # normalize = matplotlib.colors.Normalize(vmin=np.nanmin(surface_300_norm), vmax=np.nanmax(surface_300_norm))
    ax.scatter(0, 0, marker = "*", color = "crimson")
    cbarr = ax.scatter(new_xs, new_ys, c=surface_300_norm, cmap = "magma_r")
    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    ax.set_box_aspect(1)
    bar = plt.colorbar(cbarr)
    bar.set_label("Surface Brightness (Log)", rotation=270, labelpad=15)


    if save == True:
        plt.savefig(name+".png", dpi = 700, bbox_inches='tight')

    return cbarr, ax

x_p = np.linspace(-10,10,700)
y_p = np.linspace(-10,10,700)

location_bulb_1 = [430, 600]
# location_bulb_2 = [122, 342] radii 3 its cool! do later
location_bulb_2 = [184, 322]


r0ly_1 = np.array(2) * fc.pctoly # radii of the dust sphere
r0ly_2 = np.array(4) * fc.pctoly # radii of the dust sphere


# r0ly = 
x_inter_1, y_inter_1, z_inter_1, angl_1, x_p1, y_p1, z_p1, z_p12 = calc_intersection_xz_plane(x_p, y_p, r0ly_1, vc.ct, location_bulb_1)
x_inter_2, y_inter_2, z_inter_2, angl_2, x_p2, y_p2, z_p2, z_p22 = calc_intersection_xz_plane(x_p, y_p, r0ly_2, vc.ct, location_bulb_2)


# x_p, y_p = np.meshgrid(x_p, y_p)
# z_p = (x_p**2 + y_p**2 - vc.ct**2) / (2 * vc.ct)
# h_1 = x_p[location_bulb_1[0], location_bulb_1[1]]
# k_1 = y_p[location_bulb_1[0], location_bulb_1[1]]
# l_1 = z_p[location_bulb_1[0], location_bulb_1[1]]

# h_2 = x_p[location_bulb_2[0], location_bulb_2[1]]
# k_2 = y_p[location_bulb_2[0], location_bulb_2[1]]
# l_2 = z_p[location_bulb_2[0], location_bulb_2[1]]
# # print(h,k,l)

# plt.figure()
# plt.contourf(x_p1, y_p1, z_p1, levels = 40, cmap='magma', alpha=0.5)
# plt.contourf(x_p1, y_p1, z_p12, levels = 40, cmap='magma', alpha=0.5)

# plt.contourf(x_p2, y_p2, z_p2, levels = 40, cmap='magma', alpha=0.5)
# plt.contourf(x_p2, y_p2, z_p22, levels = 40, cmap='magma', alpha=0.5)

# plt.scatter(x_inter_1, y_inter_1, alpha = 0.1)
# plt.scatter(x_inter_2, y_inter_2, alpha = 0.1)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Centers at x0 = [%.2f, %.2f], y0 = [%.2f, %.2f], z0 = [%.2f, %.2f]'%(h_1,h_2,k_1,k_2,l_1,l_2))
# plt.grid(True)
# plt.axis('equal')
# name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"sphere_ncenter_countour_r%s.pdf"%([r0ly_1,
#                                                                                                                                           r0ly_2,location_bulb_1,
#                                                                                                                                           location_bulb_2])
# # Show the plot
# plt.savefig(name, dpi = 700)
# plt.show()

angl = np.concatenate((angl_1, angl_2))
x_inter = np.concatenate((x_inter_1, x_inter_2))
y_inter = np.concatenate((y_inter_1, y_inter_2))
z_inter = np.concatenate((z_inter_1, z_inter_2))

cossigma_1, surface_1 = sb.surface_brightness(x_inter_1, y_inter_1, z_inter_1, vc.ct)
cossigma_2, surface_2 = sb.surface_brightness(x_inter_2, y_inter_2, z_inter_2, vc.ct)

surface = np.concatenate((surface_1, surface_2))
cossigma = np.concatenate((cossigma_1, cossigma_2))


phis, new_xs, new_ys = rinout_blub(x_inter, y_inter, z_inter)
r0ly1 = np.round(r0ly_1, 2)
r0ly2 = np.round(r0ly_2, 2)
path_ge = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\num\\"
np.save(path_ge+r"sphere_2bulb_xs_r%s.npy"%([[r0ly1, r0ly2], [location_bulb_1, location_bulb_2]]), new_xs)
np.save(path_ge+r"sphere_2bulb_ys_r%s.npy"%([[r0ly1, r0ly2], [location_bulb_1, location_bulb_2]]), new_ys)
np.save(path_ge+r"sphere_2bulb_surface_r%s.npy"%([[r0ly1, r0ly2], [location_bulb_1, location_bulb_2]]), surface)
np.save(path_ge+r"sphere_2bulb_delta_r%s.npy"%([[r0ly1, r0ly2], [location_bulb_1, location_bulb_2]]), angl)
np.save(path_ge+r"sphere_2bulb_cossigma_r%s.npy"%([[r0ly1, r0ly2], [location_bulb_1, location_bulb_2]]), cossigma)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot(surface, new_xs, new_ys, ax, fig, save = False, name = "name")

ax.set_title(r"V838 Mon - 2Blub: time$_{obs}$ = %s days, r0 = %s ly, location = %s index"%(vc.Deltat, [r0ly1, r0ly2], [location_bulb_1, location_bulb_2]))
name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"sphere_2bulb_v838_r%s.pdf"%([[r0ly1, r0ly2], [location_bulb_1, location_bulb_2]])
plt.savefig(name, dpi = 700)
plt.show()