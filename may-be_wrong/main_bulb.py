import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
import astropy.cosmology.units as cu
from astropy.cosmology import FlatLambdaCDM
from scipy.special import erf
from scipy import integrate


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
import brightness as fb



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
    name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"sphere_ncenter_countour_r%s.pdf"%([r0ly, location_bulb])
    # Show the plot
    plt.savefig(name, dpi = 700)
    # plt.show()

    # calculate the angle between the line of sight (z) and the vector source-dust (x,y,z)
    angl = np.arccos(z_inter / np.sqrt(x_inter**2 + y_inter**2 + z_inter**2))

    return x_inter, y_inter, z_inter, angl


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

def brightness(x_inter, y_inter, z_inter, ct):
    """
    Calculate the fluxxy brightness at a position r = (x_inter, y_inter, z_inter): 
    Sugermann 2003 equation 7:
        SB(lambda, t) = F(lambda)nH(r) * (c dz0 / (4 pi r rhodrho) )* S(lambda, mu) 
        S(lambda, mu) = \int Q(lamdda, a) sigma Phi(mu, lambda, a) f(a) da
        lambda: given wavelength in micrometer [lenght]
        dz0: dust thickness [lenght]
        r: position dust [lenght]
        rhodrho: x-y of LE [lenght^2]
        mu: cos theta, theta: scattering angle
        Q: albedo
        sigma: cross section [lenght^2]
        Phi: scattering function
        f(a): dust distribution [1/lenght]
        S: scattering integral [lenght^2]

    Arguments:
        x_inter, y_inter, z_inter: intersection paraboloid + dust in ly
        dz0: thickness dust in ly
        ct: time where the LE is observed in y

    Return
        fluxxy brightness in units of kg/(s^3 l^2) 
        cos(scatter angle)
    """
        

    # m_peak = vc.m_peak
    # M_sun = fc.Msun
    # F_10_sun = 3.194e-17 # erg/scm2 at 10pc
    # F = F_10_sun*10**((m_peak - M_sun) / 2.5)
    
    # Sugerman 2003 after eq 15 F(lambda) = 1.25*F(lambda, tmax)*0.5*dt0
    F = dc.Flmax #1.08e-14 # watts / m2
    F = F * (fc.ytos**3) # kg,ly,y
    # Ir = 1.25*F*0.5*vc.dt0 * fc.n_H * fc.c
    Ir = F * fc.n_H 

    # calculate r, source-dust
    r = np.sqrt(x_inter**2 + y_inter**2 + z_inter**2)

    # calculate rho, x-y projection
    # rhos = np.sqrt(2 * z_inter * ct + (ct)**2 )
    # thickness sugermann 2003 eq 11
    # half_obs_thickness = np.sqrt( (ct / rhos) ** 2 * vc.dz0 ** 2 + ( (rhos * fc.c / 2 * ct) + ( fc.c * ct / 2 * rhos )) ** 2 * vc.dt0  ** 2 )
    # rhodrho = rhos * half_obs_thickness

    # dust-observer
    ll = np.sqrt(x_inter**2 + y_inter**2 + (z_inter-vc.d)**2)
    # calcualte scatter angle, angle between source-dust , dust-observer
    cossigma = ((x_inter**2 + y_inter**2 + z_inter * (z_inter-vc.d)) / (r * ll))

    # Calculate the scattering integral and the fluxxy brightness
    S = np.zeros(len(r))
    for ik, rm in enumerate(cossigma):
        if ((rm >= -1) and (rm <= 1)):
            ds, Scm = csf.main(rm, wave = dc.wavel) # 1.259E+00 in um
            # print(Scm)
            S[ik] = (Scm[0] * fc.pctoly**3) / (100 * fc.pctom )**3 # conver to ly
        else:
            S[ik] = 0
    # fluxxy = np.zeros(len(r))
    # for ff in range(len(x_inter)):
    Inte = integrate.simpson(1/r**2, z_inter)
    fluxxy = Ir * S * Inte / ( 4 * np.pi )


    return cossigma, fluxxy




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

x_p = np.linspace(-10,10,1000)
y_p = np.linspace(-10,10,1000)

location_bulb = [430, 700]

# r0ly = 
x_inter, y_inter, z_inter, angl = calc_intersection_xz_plane(x_p, y_p, vc.r0ly, vc.ct, location_bulb)
# cossigma, surface = sb.surface_brightness(x_inter, y_inter, z_inter, vc.ct)
# cossigma, flux = fb.brightness(x_inter, y_inter, z_inter, vc.ct)
cossigma, flux = fb.brightness(x_inter, y_inter, z_inter, vc.ct)


phis, new_xs, new_ys = rinout_blub(x_inter, y_inter, z_inter)
path_ge = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\num\\"
np.save(path_ge+r"sphere_bulb_xs_r%s.npy"%([vc.r0ly, location_bulb]), new_xs)
np.save(path_ge+r"sphere_bulb_ys_r%s.npy"%([vc.r0ly, location_bulb]), new_ys)
np.save(path_ge+r"sphere_bulb_flux_r%s.npy"%([vc.r0ly, location_bulb]), flux)
np.save(path_ge+r"sphere_bulb_delta_r%s.npy"%([vc.r0ly, location_bulb]), angl)
np.save(path_ge+r"sphere_bulb_cossigma_r%s.npy"%([vc.r0ly, location_bulb]), cossigma)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot(flux, new_xs, new_ys, ax, fig, save = False, name = "name")
ax.set_title(r"V838 Mon - Blub: time$_{obs}$ = %s days, r0 = %s ly, location = %s index"%(vc.Deltat, vc.r0ly, location_bulb))
name = r"C:\Users\\tac19\\OneDrive\\Documents\\UDEL\\Project_RA\\LE\\Simulation\\results\\figures\\"+r"sphere_bulb_v838_r%s.pdf"%([vc.r0ly, location_bulb])
plt.savefig(name, dpi = 700)
plt.show()