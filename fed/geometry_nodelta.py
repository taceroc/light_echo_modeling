import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
import astropy.cosmology.units as cu
from astropy.cosmology import FlatLambdaCDM

import sys
from setpath import path_to_LE
# sys.path.append('/content/drive/MyDrive/LE2023/dust/code')
sys.path.append(path_to_LE + r"/dust/code")

import var_constants as vc
import dust_constants as dc
import fix_constants as fc
import scattering_function as sf
import size_dist as sd
import calculate_scattering_function as csf

sys.path.append(path_to_LE)
import surface_brightness as sb
# import brightness as fb



plt.style.use('seaborn-v0_8-colorblind')

def calc_intersection_xz_plane(x, z0ly, a, ct):
    """
    Calculate the intersection points x,y,z between the plane and the paraboloid

    Arguments:
        x: initialize values for x, e.g: x = np.linspace(-10, 10, 1000) in ly
        z0ly: plane intersects the line of sight here in ly
        a: inclination of the plane a = tan(alpha)
        ct: time where the LE is observed

    Return:
        x_inter, y_inter, z_inter: intersection plane and paraboloid
        angl: angle between line of sight and source-dust that are inside the delta range
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
    print(y_inter_values.shape)


    # extract x where y is no nan
    x_inv_nan = np.hstack((x, x.copy()))
    x_inter_values = x_inv_nan[~np.isnan(y_inter)]
    print(x_inter_values.shape)

    # calculate z = z0 - ax >> plane equation
    z_inter_values = z0ly - a * x_inter_values

    # calculate the angle between the line of sight (z) and the vector source-dust (x,y,z)
    angl = np.arccos(z_inter_values / np.sqrt(x_inter_values**2 + y_inter_values**2 + z_inter_values**2))

    return x_inter_values, y_inter_values, z_inter_values, angl


def calc_intersection_xz_sphere(x, r0ly, ct):
    """
    Calculate the intersection points x,y,z between a sphere center at the source and the paraboloid

    Arguments:
        x: initialize values for x, e.g: x = np.linspace(-10, 10, 1000) in ly
        r0ly: radii of sphere in ly
        ct: time where the LE is observed

    Return:
        x_inter, y_inter, z_inter: intersection sphere and paraboloid
        angl: angle between line of sight and source-dust that are inside the delta range
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

    return x_inter_values, y_inter_values, z_inter_values, angl

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
    phis = np.arctan2(y_inter, x_inter + a*ct)
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


def final_xy_projected_in_array(phis, inner_radius, outer_radius,
                                act, array_2d, new_xs, new_ys):

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
    # DOES NOT WORK ATM 11/20/2023

    def is_point_in_annulus(center, point, outer_radius, inner_radius, shape):
        center = np.array([int(shape[1] / 2), center])
        distance = np.linalg.norm(np.array(point) - np.array(center))
        #print(point, center, distance, inner_radius, outer_radius)
        return inner_radius <= distance <= outer_radius

    def is_element_center_in_annulus(array, row, col, center,
                                    inner_radius, outer_radius):
        element_center = (row + 0.5, col + 0.5)  # Assuming array indices start from 0


        return is_point_in_annulus(center, element_center,
                                   inner_radius, outer_radius, array.shape)

    mins = np.min((new_xs, new_ys))
    maxs = np.max((new_xs, new_ys))
    print(array_2d.shape)
    for i in range(array_2d.shape[0]):
       for j in range(array_2d.shape[1]):
            x, y = (i / array_2d.shape[0]) * (maxs-mins) + mins, (j / array_2d.shape[1]) * (588*2) + mins
            ## NEED TO REVISE ^^^
            ##print(i, j, x, y, mins, maxs, inner_radius(x))
            if is_element_center_in_annulus(array_2d, x, y, act,
                                            inner_radius(x), outer_radius(x)):
                array_2d[i, j] = 1
                # this should calculate the surface brightness

    plt.imshow(array_2d)
    plt.show()
    return array_2d



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


def LE_xy_surface_concate_plane(alpha, z0ly, ct, x):
    """
    Calculate the intersection points x,y,z between the plane and the paraboloid

    Arguments:
        x: initialize values for x, e.g: x = np.linspace(-10, 10, 1000) in ly
        z0ly: plane intersects the line of sight here in ly
        a: inclination of the plane a = tan(alpha)
        ct: time where the LE is observed

    Return:
        new_xs, new_ys: in arcsec
        surface: 
        act: in arc
        fin_delta: angle line of sight - dust

    """
    a = np.tan(np.deg2rad(alpha))
    r_le2 = 2 * z0ly * ct + (ct)**2 * (1 + a**2)
    r_le = np.sqrt(r_le2)

    def calculation(alpha, z0ly, ct, a, r_le2, r_le, x):
        a = np.tan(np.deg2rad(alpha))

        x_inter, y_inter, z_inter, ange = calc_intersection_xz_plane(x, z0ly, a, ct)
        
        cossigma, surface = sb.surface_brightness(x_inter, y_inter, z_inter, ct)
        # cossigma, surface = fb.brightness(x_inter, y_inter, z_inter, ct)

        phis, r_le_out, r_le_in, act = rinout_plane(y_inter, x_inter, ct, a, z0ly)
        print(phis, r_le_out, r_le_in)
        plt.plot(x_inter, r_le_in, '.')
        plt.show()
        print("x_inter", x_inter.shape, r_le_out.shape)
        r_le_out_f = sp.interpolate.interp1d(x_inter, r_le_out, fill_value=(r_le_out[0], r_le_out[-1]))
        r_le_in_f = sp.interpolate.interp1d(x_inter, r_le_in, fill_value=(r_le_in[0], r_le_in[-1]))
        new_xs, new_ys = final_xy_projected(phis, r_le_out, r_le_in, act)

        #final_xy_projected_in_array(phis, r_le_out_f, r_le_in_f, act,
        #                            np.zeros((100, 100)), new_xs, new_ys)

        return new_xs, new_ys, surface, act, ange, cossigma


    if z0ly < 0:
        ti = (-2 * z0ly)/(fc.c * (1 + a**2))
        if ti >= ct:
            print("No LE")
            return 0,0,0,0,0,0
        else:
            new_xs, new_ys, surface, act, ange, cossigma = calculation(alpha, z0ly, ct, a, r_le2, r_le, x)
            return new_xs, new_ys, surface, act, ange, cossigma
    else:
        new_xs, new_ys, surface, act, ange, cossigma = calculation(alpha, z0ly, ct, a, r_le2, r_le, x)
        return new_xs, new_ys, surface, act, ange, cossigma
    



def LE_xy_surface_concate_plane_fed(alpha, z0ly, ct, x):
    """
    Calculate the intersection points x,y,z between the plane and the paraboloid

    Arguments:
        x: initialize values for x, e.g: x = np.linspace(-10, 10, 1000) in ly
        z0ly: plane intersects the line of sight here in ly
        a: inclination of the plane a = ctan(alpha)
        ct: time where the LE is observed

    Return:
        new_xs, new_ys: in arcsec
        surface: 
        act: in arc
        fin_delta: angle line of sight - dust

    """
    a = np.tan(np.deg2rad(alpha))
    r_le2 = 2 * z0ly * ct + (ct)**2 * (1 + a**2)
    r_le = np.sqrt(r_le2)

    def calculation(alpha, z0ly, ct, a, r_le2, r_le, x):
        a = np.tan(np.deg2rad(alpha))

        x_inter, y_inter, z_inter, ange = calc_intersection_xz_plane(x, z0ly, a, ct)
        
        cossigma, surface = sb.surface_brightness(x_inter, y_inter, z_inter, ct)
        # cossigma, surface = fb.brightness(x_inter, y_inter, z_inter, ct)

        phis, r_le_out, r_le_in, act = rinout_plane(y_inter, x_inter, ct, a, z0ly)

        new_xs, new_ys = final_xy_projected(phis, r_le_in, r_le_out, act)

        arr2d = np.zeros((201, 201))
        #arr2d = final_xy_projected_in_array(phis, r_le_out, r_le_in,
        #                                    act, arr2d, new_xs, new_ys)

        #import pylab as plt
        #plt.imshow(arr2d)
        #plt.show()


        #print("num surface: %s"%(surface.shape))

        return new_xs, new_ys, surface, act, ange, cossigma


    if z0ly < 0:
        ti = (-2 * z0ly)/(fc.c * (1 + a**2))
        if ti >= ct:
            print("No LE")
            return 0,0,0,0,0,0
        else:
            new_xs, new_ys, surface, act, ange, cossigma = calculation(alpha, z0ly, ct, a, r_le2, r_le, x)
            return new_xs, new_ys, surface, act, ange, cossigma
    else:
        new_xs, new_ys, surface, act, ange, cossigma = calculation(alpha, z0ly, ct, a, r_le2, r_le, x)
        return new_xs, new_ys, surface, act, ange, cossigma
    



def LE_xy_surface_concate_sphere(r0ly, ct, x):
    """
    Calculate the intersection points x,y,z between the sohere and the paraboloid

    Arguments:
        x: initialize values for x, e.g: x = np.linspace(-10, 10, 1000) in ly
        r0ly: radii dust sphere
        ct: time where the LE is observed

    Return:
        new_xs, new_ys: in arcsec
        surface: 
        fin_delta: angle line of sight dust

    """
    r_le2 = 2 * r0ly * ct - (ct)**2 
    r_le = np.sqrt(r_le2)

    def calculation(r0ly, ct, r_le2, r_le, x):

        x_inter, y_inter, z_inter, ange = calc_intersection_xz_sphere(x, r0ly, ct)
        
        cossigma, surface = sb.surface_brightness(x_inter, y_inter, z_inter, ct)
        # cossigma, surface = fb.brightness(x_inter, y_inter, z_inter, ct)

        phis, r_le_out, r_le_in = rinout_sphere(x_inter, y_inter, z_inter, ct, r0ly)
        new_xs, new_ys = final_xy_projected_sphere(phis, r_le_out, r_le_in)

        print("num surface: %s"%(surface.shape))

        return new_xs, new_ys, surface, ange, cossigma


    if r0ly > 0:
        new_xs, new_ys, surface, ange, cossigma = calculation(r0ly, ct, r_le2, r_le, x)
        return new_xs, new_ys, surface, ange, cossigma
    else:
        print("No LE")
        return 0,0,0,0,0

