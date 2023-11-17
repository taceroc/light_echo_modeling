import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import sys
sys.path.append(r"path_to_LE/dust/code")

import var_constants as vc
import dust_constants as dc
import fix_constants as fc
import scattering_function as sf
import size_dist as sd
import calculate_scattering_function as csf



def surface_brightness(x_inter, y_inter, z_inter, ct):
    """
    Calculate the surface brightness at a position r = (x_inter, y_inter, z_inter): 
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
        Surface brightness in units of kg/ly^3 [erg/(s cm4)]
        cos(scatter angle)
    """
        

    # m_peak = vc.m_peak
    # M_sun = fc.Msun
    # F_10_sun = 3.194e-17 # erg/scm2 at 10pc
    # F = F_10_sun*10**((m_peak - M_sun) / 2.5)
    
    # Sugerman 2003 after eq 15 F(lambda) = 1.25*F(lambda, tmax)*0.5*dt0
    F = dc.Flmax #1.08e-14 # watts / m2
    F = F * (fc.ytos**3) # kg,ly,y
    Ir = 1.25*F*0.5*vc.dt0 * fc.n_H * fc.c

    # calculate r, source-dust
    r = np.sqrt(x_inter**2 + y_inter**2 + z_inter**2)

    # calculate rho, x-y projection
    rhos = np.sqrt(2 * z_inter * ct + (ct)**2 )
    # thickness sugermann 2003 eq 11
    half_obs_thickness = np.sqrt( (ct / rhos) ** 2 * vc.dz0 ** 2 + ( (rhos * fc.c / 2 * ct) + ( fc.c * ct / 2 * rhos )) ** 2 * vc.dt0  ** 2 )
    rhodrho = rhos * half_obs_thickness

    # dust-observer
    ll = np.sqrt(x_inter**2 + y_inter**2 + (z_inter-vc.d)**2)
    # calcualte scatter angle, angle between source-dust , dust-observer
    cossigma = ((x_inter**2 + y_inter**2 + z_inter * (z_inter-vc.d)) / (r * ll))

    # Calculate the scattering integral and the surface brightness
    S = np.zeros(len(r))
    for ik, rm in enumerate(cossigma):
        if ((rm >= -1) and (rm <= 1)):
            ds, Scm = csf.main(rm, wave = dc.wavel) # 1.259E+00 in um
            # print(Scm)
            S[ik] = (Scm[0] * fc.pctoly**2) / (100 * fc.pctom )**2 # conver to ly
        else:
            S[ik] = 0
    surface = np.zeros(len(r))
    for ff in range(len(x_inter)):
        surface[ff] = Ir * S[ff] * vc.dz0 / ( 4 * np.pi * r[ff] * rhodrho[ff] )


    return cossigma, surface
