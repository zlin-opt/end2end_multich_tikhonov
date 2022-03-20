import numpy as np
import mpi4py.MPI as MPI
comm=MPI.COMM_WORLD

nwavs=16
wavmin=470.
wavmax=660.
recpwavs=np.linspace(1/wavmin,1/wavmax,nwavs)
wavs=1./recpwavs
wavcen=np.round(1./np.mean(recpwavs),2)
freqs=wavcen/wavs

tio2dat=np.loadtxt('tio2_refractiveindex_[nm].txt')
epsTiO2=np.interp(wavs,tio2dat[:,0],tio2dat[:,1])**2
def mat_sio2(lamnm):
    lam=lamnm/1000.0
    return 1 + 0.6961663*lam**2/(lam**2-0.0684043**2) + 0.4079426*lam**2/(lam**2-0.1162414**2) + 0.8974794*lam**2/(lam**2-9.896161**2)
epsSiO2=mat_sio2(wavs)
###

unitcell_size = np.round(0.99 * wavmin,2)

minfea = 60.
maxfea = unitcell_size-minfea
order = 1279

filename='chebptsord{}.txt'.format(order)

thickness=600.

zmin=2e7/wavcen
zmax=2e7/wavcen
ndepths=1
depths=np.linspace(zmin,zmax,ndepths)

ntruth=50
nextra=50
npad=0
nsr=1e3
tol=1e-16
nintg=3
du=np.round(unitcell_size/wavcen,2)
Dz =2e3
ngeom=1
lbgeom=[np.round(minfea/wavcen,2)]
ubgeom=[np.round(maxfea/wavcen,2)]
scale=1.

###

nfreqs=freqs.size
nch=nfreqs*ndepths
nsens=ntruth+nextra
nshot=int(np.ceil(np.sqrt(nch)))*nsens+npad
npsf=nshot+ntruth
nfar=npsf*nintg
ncells=npsf*nintg
ngreen=ncells+nfar

outputfile = 'alldat_tio2_{}nm_order{}_{}channels.dat'.format(thickness,order,nch)

pixelsize=nintg*du*wavcen/1000. #[um]
pbsendist=Dz*wavcen/1000. #[um]
resang=np.arctan(pixelsize/pbsendist)
viewang=np.arctan(ntruth*pixelsize/pbsendist)
objsize=2.*np.min(depths)*np.tan(viewang/2.)*wavcen/1e6
NA = np.sin(np.arctan(ncells*du/(2.*Dz)))
fwhm = np.round(np.max(wavs)/(2.*NA),2)/1000.

if comm.rank==0:
    print("=======****INFO****=======")
    print("{} wavelength channels: {} nm, center {} nm".format(nfreqs,np.round(wavs,2),wavcen))
    print("{} freqs channels: {} (MEEP units)".format(nfreqs,np.round(freqs,2)))
    print("{} wavelength channels: {} (MEEP units)".format(nfreqs,np.round(1./freqs,2)))
    print("unitcell size: {} (MEEP units), or {} nm; min wavlen: {}".format(du,unitcell_size,wavmin/wavcen))
    print("min and max features: {}, {} nm".format(minfea,maxfea))
    print("lbgeom {} and ubgeom {} (MEEP units)".format(lbgeom,ubgeom))
    print("minimum gap: {} nm".format(unitcell_size - maxfea))

    print("=======")
    print("{} depths channels: {} mm".format(ndepths,np.round(depths*wavcen/1e6,2)))
    print("detector pixel spans {} grid-points (per dimension) or {} um".format(nintg,pixelsize))
    print("probe-to-sensor distance: {}x{} nm or {} mm".format(Dz,wavcen,pbsendist/1e3))
    print("resolution angle (detector side) is {} degree".format(resang*180/np.pi))
    print("view angle (object side) is {} degree".format(viewang*180/np.pi))
    print("Object size: {} mm".format(np.round(objsize,2)))
    print("numerical aperture: {}, fwhm: {} um".format(NA,fwhm))
        
    print("=======")
    print('interp filename: {}'.format(outputfile))
    print("... object resolution: {}^2 pixels x {} channels, sub-image size: {}^2 pixels, full image size: {}^2 pixels, PSF size: {}^2 pixels  [[detector pixels]]".format(ntruth,nch,nsens,nshot,npsf))
    print("... PSF grid-size: {}^2, ncells: {}^2, far-field kernel grid-size: {}^2".format(nfar,ncells,ngreen))
    print("... (in units of {} nm) PSF grid-size: {}^2 or {}^2 mm^2, device-size: {}^2 or {}^2 mm^2".format(wavcen,nfar*du,nfar*du*wavcen/1e6,ncells*du,ncells*du*wavcen/1e6))
    print("nsr: {}".format(nsr))
    
comm.Barrier()

