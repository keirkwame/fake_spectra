"""File to find the temperature density relation of the forest
and make a temperature density plot weighted by HI fraction.

Main function is fit_td_rel_plot()"""

import numpy as np
from scipy.optimize import leastsq
#import matplotlib
from . import abstractsnapshot as absn
from . import unitsystem as units
from .gas_properties import GasProperties
#from .ratenetworkspectra import RateNetworkGas
#matplotlib.use("PDF")
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
import bigfile

def mean_density(hub, redshift, omegab=0.0465):
    """Get mean gas density at some redshift."""
    unit = units.UnitSystem()
    #in g cm^-3
    rhoc = unit.rho_crit(hub)

    #Convert to atoms per cm^-3
    rhoc /= unit.protonmass

    nH = rhoc * omegab * (1 + redshift)**3

    return nH

def fit_temp_dens_relation(logoverden, logT):
    """Fit a temperature density relation."""
    
    ind = np.where((0.1 < logoverden) * (logoverden <  1.0) * (0.1 < logT) * (logT < 5.0))

    logofor = logoverden[ind]
    logtfor = logT[ind]

    def min_func(param):
        """Function to minimize: power law fit to temperature density relation."""
        logT0 = param[0]
        gammam1 = param[1]
        #print(param)
        return logtfor - (logT0 + gammam1 * logofor)
    res = leastsq(min_func, np.array([np.log10(1e4), 0.5]), full_output=True)
    params = res[0]
    if res[-1] <= 0:
        print(res[3])
    return 10**params[0], params[1] + 1
'''
def fit_td_rel_plot(num, base, nhi=True, nbins=500, gas="raw", plot=True,Tscale=1, gammascale=1,printOutput=False):
    """Make a temperature density plot of neutral hydrogen or gas.
    Also fit a temperature-density relation for the total gas (not HI).
    Arguments:
        num - snapshot number
        base - snapshot base directory
        nbins - number of bins to use for the T-rho histogram
        gas - if "raw" use snapshot values for temperature and neutral fraction. Otherwise use rate network values.
        nhi - if True, plot neutral hydrogen, otherwise plot total gas density
        plot - if True, make a plot, otherwise just do the fit
    """
    print(num)
    print(base)
    snap = absn.AbstractSnapshotFactory(num, base, Tscale, gammascale)

    redshift = 1./snap.get_header_attr("Time") - 1
    hubble = snap.get_header_attr("HubbleParam")
    if gas == "raw":
        rates = GasProperties(redshift, snap, hubble)
    else:
        rates = RateNetworkGas(redshift, snap, hubble)

    temp = rates.get_temp(0, -1)

    dens = rates.get_code_rhoH(0, -1)

    logdens = np.log10(dens)
    logT = np.log10(temp)
    mean_dens = mean_density(hubble, redshift, omegab=snap.get_omega_baryon())
    (T0, gamma) = fit_temp_dens_relation(logdens - np.log10(mean_dens), logT)
    if printOutput==True:
        print("z=%f T0(K) = %f, gamma = %g" % (redshift, T0, gamma))
    del snap
    if plot:
        if nhi:
            nhi = rates.get_reproc_HI(0, -1)
        else:
            nhi = dens

        hist, dedges, tedges = np.histogram2d(logdens, logT, bins=nbins, weights=nhi, density=True)

        plt.imshow(hist.T, interpolation='nearest', origin='low', extent=[dedges[0], dedges[-1], tedges[0], tedges[-1]], cmap=plt.cm.cubehelix_r, vmax=0.75, vmin=0.01)

        plt.plot(np.log10(mean_dens), np.log10(T0), '*', markersize=10, color="gold")
        dd = np.array([-6,-5,-4,-3])
        plt.xticks(dd, [r"$10^{%d}$" % d for d in dd])
        tt = np.array([2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000])
        plt.yticks(np.log10(tt), tt//1000)
        plt.ylabel(r"T ($10^3$ K)")
        plt.xlabel(r"$\rho$ (cm$^{-3}$)")

        plt.xlim(-6,-3)
        plt.ylim(3.4,5)
        plt.colorbar()
        plt.tight_layout()
    return T0, gamma
'''
def fit_td_rel_plot(num, base, Tscale=1, gammascale=1, plot=False):
    Nbins=1000 ## Number of bins along each axis
    maxOverDensity=0.5 ## Maximum overdensity to use for T_0 and gamma plot
    snap=str(num).rjust(3,'0')
    snapshot=os.path.join(base, "PART_"+snap)
    ## Factores required to calculate temperature
    gamma=5./3
    hy_mass=0.76
    protonmass=1.67262178e-24 # proton mass in g
    boltzmann=1.38066e-16 # k_b in cgs
    UnitVelocity_in_cm_per_s=1e5
    UnitInternalEnergy_in_cgs = UnitVelocity_in_cm_per_s**2

    f = bigfile.File(snapshot)
    ## Get simulation params from header
    boxSize=(f["Header"].attrs["BoxSize"][0])/1000
    z=(1/f["Header"].attrs["Time"][0])-1
    NPart=round((f["Header"].attrs["TotNumPartInit"][0])**0.33333333)
    om_b=f["Header"].attrs["OmegaBaryon"]

    ## Calculate mean density
    rho_crit=27.75e-9
    mean_baryon_dens=rho_crit*om_b

    ## Gas = 0, DM = 1
    density=bigfile.Dataset(f["0/"], ['Density'])
    ienergy=bigfile.Dataset(f["0/"], ['InternalEnergy'])
    nelec=bigfile.Dataset(f["0/"] , ['ElectronAbundance'])
    ## Convert to array
    density=density[:].astype(float)
    ienergy=ienergy[:].astype(float)
    nelec=nelec[:].astype(float)

    ## Temp rescaling if required
    if Tscale!=1:
        ienergy*=Tscale
    if gammascale!=1:
        ienergy*=(density/mean_baryon_dens)**(gammascale-1)

    muienergy=4/(hy_mass*(3+4*nelec)+1)*(ienergy*UnitInternalEnergy_in_cgs)
    #So for T in K, boltzmann in erg/K, internal energy has units of erg/g
    temp=(gamma-1)*protonmass/boltzmann*muienergy

    ## Convert density into overdensity, and take logs
    density=np.log10(density/mean_baryon_dens)
    temp=np.log10(temp)

    ## Now plot temperature density relation
    if plot==True:
        plt.figure()
        plt.title("z=%.1f, box=%.1f Mpc/h, Npart=%d" % (z,boxSize,NPart))
        plt.xlabel(r"$\mathrm{log}_{10}(\rho_b/\bar{\rho}_b)$")
        plt.ylabel(r"$\mathrm{log}_{10}$ Temperature (K)")

    Data2D,xedges,yedges,quad=plt.hist2d(density,temp,bins=Nbins,cmap="jet",norm=LogNorm())

    ## Next step is to fit the temperature density profile
    aa=0
    tempFit=np.array([])
    while xedges[aa]<=maxOverDensity:
        ## Get argument of median
        argmed=(np.abs(Data2D[aa][:] - np.median(np.trim_zeros(Data2D[aa][:])))).argmin()
        tempFit=np.append(tempFit,yedges[argmed])   # Median fit
        print("Median arg=", argmed)
        print("Mode arg=", np.argmax(Data2D[aa][:]))
        #tempFit=np.append(tempFit,yedges[np.argmax(Data2D[aa][:])])   # Mode fit
        aa=aa+1
    gamma_minus_one,logT0=np.polyfit(xedges[:aa],tempFit,deg=1)
    #print("T0=", 10**logT0, "\n gamma", gamma_minus_one+1)
    fitLine=xedges[np.where(xedges<1.)[0]]*gamma_minus_one+logT0
    if plot==True:
        ## Add TDR fit to plot
        plt.colorbar()
        plt.plot(xedges[np.where(xedges<1.)[0]],fitLine,color="black",linestyle="dashed")
        plt.ylim(3,5)
        plt.text(min(xedges)+0.1,max(yedges)-0.25,r"$T_0=%.2e$" % (10**logT0))
        plt.text(min(xedges)+0.1,max(yedges)-0.45,r"$\gamma=%.2f$" % (gamma_minus_one+1))
        plt.show("hold")
    return 10**logT0, gamma_minus_one+1


