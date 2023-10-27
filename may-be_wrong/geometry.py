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
import brightness as fb
import surface_brightness as sb



plt.style.use('seaborn-v0_8-colorblind')

def calc_intersection_xz_plane(x, delta, z0ly, a, ct):
    """
    Calculate the intersection points x,y,z between the plane and the paraboloid

    Arguments:
        x: initialize values for x, e.g: x = np.linspace(-10, 10, 1000) in ly
        delta: angle(s) where the dust is valid in deg, angle between z and r
        z0ly: plane intersects the line of sight here in ly
        a: inclination of the plane a = tan(alpha)
        ct: time where the LE is observed

    Return:
        x_inter, y_inter, z_inter: intersection plane and paraboloid
        angl: angle between line of sight and source-dust that are inside the delta range
        indexs: indexs where the angle are valid
    """

    # Intersection paraboloid and plane give the LE radii
    r_le2 = 2 * z0ly * ct + (ct)**2 * (1 + a**2)
    r_le = np.sqrt(r_le2)

    # (x + act)^2 + y^2 = rle^2 --> y12 = +-sqrt(rle^2 - (x + act)^2)
    # calculate the y 
    y_1 = np.sqrt(r_le2 - (x + a * ct)**2)
    y_2 = -1*y_1

    # keep no nan values
    y_inter = np.hstack((y_1, y_2))
    y_inter_values = y_inter[~np.isnan(y_inter)]

    # extract x where y is no nan
    x_inv_nan = np.hstack((x, x.copy()))
    x_inter_values = x_inv_nan[~np.isnan(y_inter)]

    # calculate z = z0 - ax >> plane equation
    z_inter_values = z0ly - a * x_inter_values

    # calculate the angle between the line of sight (z) and the vector source-dust (x,y,z)
    angl = np.arccos(z_inter_values / np.sqrt(x_inter_values**2 + y_inter_values**2 + z_inter_values**2))

    # given a delta angle keep only x,y,z intersection where the angle is in the delta range 
    indexs = []
    delta = np.deg2rad(delta)
    for ii, dd in enumerate(angl):
        if ((delta - np.deg2rad(5)) <= dd) & (dd <= (delta + np.deg2rad(5))):
            indexs.append(ii)

    return x_inter_values, y_inter_values, z_inter_values, angl, indexs


def calc_intersection_xz_sphere(x, delta, r0ly, ct):
    """
    Calculate the intersection points x,y,z between a sphere center at the source and the paraboloid

    Arguments:
        x: initialize values for x, e.g: x = np.linspace(-10, 10, 1000) in ly
        delta: angle(s) where the dust is valid in deg, angle between z and r
        r0ly: radii of sphere in ly
        ct: time where the LE is observed

    Return:
        x_inter, y_inter, z_inter: intersection sphere and paraboloid
        angl: angle between line of sight and source-dust that are inside the delta range
        indexs: indexs where the angle are valid
    """

    # Intersection paraboloid and sphere give the LE radii
    r_le2 = 2 * r0ly * ct - (ct)**2 
    r_le = np.sqrt(r_le2)

    # x^2 + y^2 = rle^2 --> y12 = +-sqrt(rle^2 - x^2)
    # calculate the y 
    y_1 = np.sqrt(r_le2 - x**2)
    y_2 = -1*y_1

    # keep no nan values
    y_inter = np.hstack((y_1, y_2))
    y_inter_values = y_inter[~np.isnan(y_inter)]

    # extract x where y is no nan
    x_inv_nan = np.hstack((x, x.copy()))
    x_inter_values = x_inv_nan[~np.isnan(y_inter)]

    # calculate z = z0 - ax >> plane equation
    z_inter_values = np.sqrt(r0ly**2 - x_inter_values**2 - y_inter_values**2)
    # z_inter_values = np.hstack((z_inter_values, -z_inter_values))

    # calculate the angle between the line of sight (z) and the vector source-dust (x,y,z)
    angl = np.arccos(z_inter_values / np.sqrt(x_inter_values**2 + y_inter_values**2 + z_inter_values**2))

    # given a delta angle keep only x,y,z intersection where the angle is in the delta range 

    indexs = []
    delta = np.deg2rad(delta)
    for ii, dd in enumerate(angl):
        if ((delta - np.deg2rad(5)) <= dd) & (dd <= (delta + np.deg2rad(5))):
            indexs.append(ii)

    return x_inter_values, y_inter_values, z_inter_values, angl, indexs

def rinout_plane(y_inter, x_inter, ct, a, z0ly):
    """
    Calculate the inner and outer radii of the LE given the thickness eq 11 Sugerman 2003
    Only valid when the dust and the paraboloid have a analytical expresion (and the analtyical expression is a circumference)

    Arguments:
        x_inter, y_inter, z_inter: intersection paraboloid + dust in ly
        ct: time where the LE is observed in y
        a: inclination of the plane a = tan(alpha)
        z0ly: plane intersects the line of sight here in ly
    
    Return:
        Phis: angle in the sky plane
        r_le_out, r_le_in: out and inner radii in arcsec
        act: center of LE in arcsec

    """
    # import this fancy thing to convert ly to arcsec, almost the same as using  np.arctan(r_le_out / vc.d) * (180 / np.pi) * 3600
    cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)
    d = (vc.d * u.lyr).to(u.Mpc)
    reds = d.to(cu.redshift, cu.redshift_distance(cosmo, kind="comoving"))
    # linear size = angular_size * d_A
    d_A = cosmo.angular_diameter_distance(z=reds)

    r_le2 = 2 * z0ly * ct + (ct)**2 * (1 + a**2)
    r_le = np.sqrt(r_le2)

    # calculate the angle in the sky plane
    phis = np.arctan2(y_inter, x_inter)
    # calculate rho, x-y projection    
    rhos = np.sqrt(2 * z0ly * ct + (ct)**2 - 2 * a * x_inter * ct)
    half_obs_thickness = np.sqrt( (ct / rhos) ** 2 * vc.dz0 ** 2 + ( (rhos * fc.c / 2 * ct) + ( fc.c * ct / 2 * rhos )) ** 2 * vc.dt0  ** 2 ) / 2
    # -- include the thickness in xy plane
    r_le_out = r_le + half_obs_thickness
    # -- degree to arcseconds
    r_le_out = (r_le_out * u.lyr).to(u.Mpc)
    # distance_Mpc = d_A * theta_radian
    r_le_out = ( r_le_out / d_A ).value / (np.pi / 180 / 3600)

    r_le_in = r_le - half_obs_thickness
    r_le_in = (r_le_in * u.lyr).to(u.Mpc)
    r_le_in = ( r_le_in / d_A ).value / (np.pi / 180 / 3600)

    act = ((a * ct) * u.lyr).to(u.Mpc)
    act = ( act / d_A ).value / (np.pi / 180 / 3600)


    return phis, r_le_out, r_le_in, act


def rinout_sphere(x_inter, y_inter, z_inter, ct, r0ly):
    """
    Calculate the inner and outer radii of the LE given the thickness eq 11 Sugerman 2003
    Only valid when the dust and the paraboloid have a analytical expresion (and the analtyical expression is a circumference)

    Arguments:
        x_inter, y_inter, z_inter: intersection paraboloid + dust in ly
        ct: time where the LE is observed in y
        r0ly: radii dust sohere in ly
    
    Return:
        Phis: angle in the sky plane
        r_le_out, r_le_in: out and inner radii in arcsec

    """
    # import this fancy thing to convert ly to arcsec, almost the same as using  np.arctan(r_le_out / vc.d) * (180 / np.pi) * 3600
    cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)
    d = (vc.d * u.lyr).to(u.Mpc)
    reds = d.to(cu.redshift, cu.redshift_distance(cosmo, kind="comoving"))
    # linear size = angular_size * d_A
    d_A = cosmo.angular_diameter_distance(z=reds)

    r_le2 = 2 * r0ly * ct - (ct)**2 
    r_le = np.sqrt(r_le2)

    # calculate the angle in the sky plane
    phis = np.arctan2(y_inter, x_inter)
    # calculate rho, x-y projection    
    rhos = np.sqrt(2 * z_inter * ct + (ct)**2 )
    half_obs_thickness = np.sqrt( (ct / rhos) ** 2 * vc.dz0 ** 2 + ( (rhos * fc.c / 2 * ct) + ( fc.c * ct / 2 * rhos )) ** 2 * vc.dt0  ** 2 ) / 2
    # -- include the thickness in xy plane
    r_le_out = r_le + half_obs_thickness
    # -- degree to arcseconds
    r_le_out = (r_le_out * u.lyr).to(u.Mpc)
    # distance_Mpc = d_A * theta_radian
    r_le_out = ( r_le_out / d_A ).value / (np.pi / 180 / 3600)

    r_le_in = r_le - half_obs_thickness
    r_le_in = (r_le_in * u.lyr).to(u.Mpc)
    r_le_in = ( r_le_in / d_A ).value / (np.pi / 180 / 3600)


    return phis, r_le_out, r_le_in



def final_xy_projected(phis, r_le_out, r_le_in, act):
    """
    Calculate the x,y points in arcseconds
    Only valid when the dust and the paraboloid have a analytical expresion (and the analtyical expression is a circumference)

    Arguments:
        phis: angle in the sky plane
        r_le_out, r_le_in: out and inner radii in arcsec
        act: center of LE in arcsec
    
    Returns:
        new_xs, new_ys: x,y position in the x-y plane in arcseconds
    """
    radii_p = [r_le_out, r_le_in]

    xs_p = np.concatenate([radii_p[0] * np.cos(phis) - act, radii_p[1] * np.cos(phis) - act]).reshape(2, len(phis))
    ys_p = np.concatenate([radii_p[0] * np.sin(phis), radii_p[1] * np.sin(phis)]).reshape(2, len(phis))

    new_xs = xs_p.reshape(1,2,len(phis))
    new_ys = ys_p.reshape(1,2,len(phis))

    return new_xs, new_ys

def final_xy_projected_sphere(phis, r_le_out, r_le_in):
    """
    Calculate the x,y points in arcseconds
    Only valid when the dust and the paraboloid have a analytical expresion (and the analtyical expression is a circumference)

    Arguments:
        phis: angle in the sky plane
        r_le_out, r_le_in: out and inner radii in arcsec
    
    Returns:
        new_xs, new_ys: x,y position in the x-y plane in arcseconds
    """
    radii_p = [r_le_out, r_le_in]

    xs_p = np.concatenate([radii_p[0] * np.cos(phis), radii_p[1] * np.cos(phis)]).reshape(2, len(phis))
    ys_p = np.concatenate([radii_p[0] * np.sin(phis), radii_p[1] * np.sin(phis)]).reshape(2, len(phis))

    new_xs = xs_p.reshape(1,2,len(phis))
    new_ys = ys_p.reshape(1,2,len(phis))

    return new_xs, new_ys


def LE_xy_surface_concate_plane(alpha, z0ly, ct, deltass, x):
    """
    Calculate the intersection points x,y,z between the plane and the paraboloid

    Arguments:
        x: initialize values for x, e.g: x = np.linspace(-10, 10, 1000) in ly
        delta: angle(s) where the dust is valid in deg, angle between z and r
        z0ly: plane intersects the line of sight here in ly
        a: inclination of the plane a = tan(alpha)
        ct: time where the LE is observed

    Return:
        new_xs, new_ys: in arcsec
        surface: 
        act: in arc
        fin_delta: delta angle valid in deg

    """
    a = np.tan(np.deg2rad(alpha))
    r_le2 = 2 * z0ly * ct + (ct)**2 * (1 + a**2)
    r_le = np.sqrt(r_le2)

    def calculation(alpha, z0ly, ct, deltass, a, r_le2, r_le, x):
        a = np.tan(np.deg2rad(alpha))
        new_xs_list = []
        new_ys_list = []

        surface_list = []

        fin_delta = []

        ange_list = []

        for deltas in deltass:
            x_inter, y_inter, z_inter, ange, indexsi = calc_intersection_xz_plane(x, deltas, z0ly, a, ct)
            # print("idne", indexsi)
            # print("xinde", x_inter[indexsi])

            x_inter = x_inter[indexsi]
            y_inter = y_inter[indexsi]
            z_inter = z_inter[indexsi]
    
            if len(indexsi) != 0:
                fin_delta.append(np.rad2deg(ange[indexsi]))
            else:
                # print("fsfs")
                fin_delta.append(np.zeros(len(ange)))
            
            cossigma, surface = fb.brightness(x_inter, y_inter, z_inter, ct)
            phis, r_le_out, r_le_in, act = rinout_plane(y_inter, x_inter, ct, a, z0ly)
            new_xs, new_ys = final_xy_projected(phis, r_le_out, r_le_in, act)

            new_xs_list.append(new_xs)
            new_ys_list.append(new_ys)
            surface_list.append(surface)
            ange_list.append(ange)

        new_xs = np.concatenate(new_xs_list, axis = 2)
        new_ys = np.concatenate(new_ys_list, axis = 2)

        flux = np.concatenate(surface_list)
        fin_delta = np.concatenate(fin_delta)

        angel = np.concatenate(ange_list)

        print("num valid angles: %s"%(fin_delta[fin_delta != 0].shape))
        print("num surface: %s"%(flux.shape))

        return new_xs, new_ys, flux, act, fin_delta, cossigma, angel


    if z0ly < 0:
        ti = (-2 * z0ly)/(fc.c * (1 + a**2))
        if ti >= ct:
            print("No LE")
            return 0,0,0,0,0,0
        else:
            new_xs, new_ys, surface, act, fin_delta, cossigma, ange = calculation(alpha, z0ly, ct, deltass, a, r_le2, r_le, x)
            return new_xs, new_ys, surface, act, fin_delta, cossigma, ange
    else:
        new_xs, new_ys, surface, act, fin_delta, cossigma, ange = calculation(alpha, z0ly, ct, deltass, a, r_le2, r_le, x)
        return new_xs, new_ys, surface, act, fin_delta, cossigma, ange
    

def LE_xy_surface_concate_sphere(r0ly, ct, deltass, x):
    """
    Calculate the intersection points x,y,z between the sohere and the paraboloid

    Arguments:
        x: initialize values for x, e.g: x = np.linspace(-10, 10, 1000) in ly
        delta: angle(s) where the dust is valid in deg, angle between z and r
        r0ly: radii dust sphere
        ct: time where the LE is observed

    Return:
        new_xs, new_ys: in arcsec
        surface: 
        fin_delta: delta angle valid in deg

    """
    r_le2 = 2 * r0ly * ct - (ct)**2 
    r_le = np.sqrt(r_le2)

    def calculation(r0ly, ct, deltass, r_le2, r_le, x):
        new_xs_list = []
        new_ys_list = []

        surface_list = []

        fin_delta = []

        for deltas in deltass:
            x_inter, y_inter, z_inter, ange, indexsi = calc_intersection_xz_sphere(x, deltas, r0ly, ct)

            x_inter = x_inter[indexsi]
            y_inter = y_inter[indexsi]
            z_inter = z_inter[indexsi]
    
            if len(indexsi) != 0:
                fin_delta.append(np.rad2deg(ange[indexsi]))
            else:
                fin_delta.append(np.zeros(len(ange)))
            
            cossigma, surface = fb.brightness(x_inter, y_inter, z_inter, ct)
            phis, r_le_out, r_le_in = rinout_sphere(x_inter, y_inter, z_inter, ct, r0ly)
            new_xs, new_ys = final_xy_projected_sphere(phis, r_le_out, r_le_in)

            new_xs_list.append(new_xs)
            new_ys_list.append(new_ys)
            surface_list.append(surface)

        new_xs = np.concatenate(new_xs_list, axis = 2)
        new_ys = np.concatenate(new_ys_list, axis = 2)

        flux = np.concatenate(surface_list)
        fin_delta = np.concatenate(fin_delta)

        print("num valid angles: %s"%(fin_delta.shape))
        print("num surface: %s"%(flux.shape))

        return new_xs, new_ys, flux, fin_delta, cossigma


    if r0ly > 0:
        new_xs, new_ys, surface, fin_delta, cossigma = calculation(r0ly, ct, deltass, r_le2, r_le, x)
        return new_xs, new_ys, surface, fin_delta, cossigma
    else:
        print("No LE")
        return 0,0,0,0,0,0



## NO DELTA DEPENDENCE

