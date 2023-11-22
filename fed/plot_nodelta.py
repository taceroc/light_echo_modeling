import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from astropy import units as u
import astropy.cosmology.units as cu
from astropy.cosmology import FlatLambdaCDM


import sys
from setpath import path_to_LE
# sys.path.append('/content/drive/MyDrive/LE2023/dust/code')
sys.path.append(path_to_LE)
sys.path.append(path_to_LE + r"/dust/code")
import var_constants as vc
import dust_constants as dc
import fix_constants as fc
import scattering_function as sf
import size_dist as sd
import calculate_scattering_function as csf

import surface_brightness as sb



plt.style.use('seaborn-v0_8-colorblind')

def setcolormap(surface, cmap = 'magma_r'):
    surface_300_norm = (surface.copy() - np.nanmin(surface.copy())) / ( # FBB why a copy?
                np.nanmax(surface.copy()) - np.nanmin(surface.copy()))
    normalize = matplotlib.colors.Normalize(vmin=np.nanmin(surface_300_norm), vmax=np.nanmax(surface_300_norm))

    return surface_300_norm, normalize

def plot(new_xs, new_ys, surface, alpha, act, ax, fig, save = False, name = "name", cmap="magma_r"):
    cmap = matplotlib.colormaps.get_cmap(cmap)
    surface_300_norm, normalize = setcolormap(surface)

    # ax.set_title("density define only for %s degrees and a = tan(%s)"%([deltass.min(), deltass.max()], alpha))

    mins = np.min((new_xs, new_ys))
    maxs = np.max((new_xs, new_ys))
    stdmin = np.min((np.std(new_xs), np.std(new_ys)))
    stdmax = np.max((np.std(new_xs), np.std(new_ys)))


    ax.set_xlim(mins - stdmin, maxs + stdmax)
    ax.set_ylim(mins - stdmin, maxs + stdmax)

    for k in range(len(surface)):
        ax.plot(new_xs[0, :, k], new_ys[0, :, k], color=cmap(normalize(surface_300_norm[k])))#, label="%s"%(z/pctoly))

    ax.scatter(- act, 0, marker = "*", color = "purple")
    ax.scatter(0, 0, marker = "*", color = "crimson")

    # cbax = fig.add_axes([0.96, 0.1, 0.03, 0.80])
    cbax = fig.add_axes([0.8, 0.1, 0.03, 0.80])


    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    ax.set_box_aspect(1)

    cb1 = matplotlib.colorbar.ColorbarBase(cbax, cmap=cmap, norm=normalize, orientation='vertical')
    cb1.set_label("Surface Brightness (Log)", rotation=270, labelpad=15)

    def label_cbrt(x,pos):
        return "{:.1f}".format(x)

    cb1.formatter = matplotlib.ticker.FuncFormatter(label_cbrt)
    # cb.update_ticks()
    # plt.tight_layout()

    if save == True:
        plt.savefig(name+".png", dpi = 700, bbox_inches='tight')

    return cb1, ax



def plot_sphere(new_xs, new_ys, surface, ax, fig, save = False, name = "name"):
    cmap = matplotlib.colormaps.get_cmap(cmap)
    surface_300_norm, normalize = setcolormap(surface)

    mins = np.min((new_xs, new_ys))
    maxs = np.max((new_xs, new_ys))
    stdmin = np.min((np.std(new_xs), np.std(new_ys)))
    stdmax = np.max((np.std(new_xs), np.std(new_ys)))


    ax.set_xlim(mins - stdmin, maxs + stdmax)
    ax.set_ylim(mins - stdmin, maxs + stdmax)

    for k in range(len(surface)):
        ax.plot(new_xs[0, :, k], new_ys[0, :, k], color=cmap(normalize(surface_300_norm[k])))#, label="%s"%(z/pctoly))

    ax.scatter(0, 0, marker = "*", color = "crimson")

    # cbax = fig.add_axes([0.96, 0.1, 0.03, 0.80])
    cbax = fig.add_axes([0.8, 0.1, 0.03, 0.80])


    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    ax.set_box_aspect(1)

    cb1 = matplotlib.colorbar.ColorbarBase(cbax, cmap=cmap, norm=normalize, orientation='vertical')
    cb1.set_label("Surface Brightness (Log)", rotation=270, labelpad=15)

    def label_cbrt(x,pos):
        return "{:.1f}".format(x)

    cb1.formatter = matplotlib.ticker.FuncFormatter(label_cbrt)
    # cb.update_ticks()
    # plt.tight_layout()

    if save == True:
        plt.savefig(name+".png", dpi = 700, bbox_inches='tight')

    return cb1, ax

def plot_2d_array(arr_2d, cmap='magma_r', center=None):
    minmax = np.nanmin(arr_2d), np.nanmax(arr_2d)
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    plt.imshow(arr_2d, cmap=cmap, clim=minmax)
    if not center:
        center = int(arr_2d.shape[0] / 2), int(arr_2d.shape[1] / 2)
    axes.scatter(center[0], center[1], marker="*", color="crimson")

    # cbax = fig.add_axes([0.96, 0.1, 0.03, 0.80])
    cbax = fig.add_axes([0.8, 0.1, 0.03, 0.80])

    axes.set_xlabel("arcsec")
    axes.set_ylabel("arcsec")
    axes.set_box_aspect(1)

    # axes.set_title(r"V838 Mon - Plane: time$_{obs}$ = %s days, $\alpha$ = %s deg,  z0 = %s pc"%(vc.Deltat, al, vc.z0))
    axes.set_title(r"V838 Mon - Plane: $\alpha$ = %s deg " % ("?"))
    name = path_to_LE + r"/figures/" + r"plane_nd_v838_a%s_heatmap.pdf" % ("?")
    plt.savefig(name, dpi=700)

    plt.show()
