import autograd.numpy as np
from autograd import grad
import n2fdiff as n2f
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD

def incident(ncells,du,freqs,depths,nsub):

    nfreqs = freqs.size
    ndepths = depths.size

    Lx = ncells*du
    Ly = ncells*du
    x,y = np.meshgrid( np.linspace(-Lx/2.+du/2.,Lx/2.-du/2.,ncells),
                       np.linspace(-Ly/2.+du/2.,Ly/2.-du/2.,ncells) )
    
    inc = np.zeros((ncells,ncells,ndepths,nfreqs),dtype=complex)
    for idepth in range(ndepths):
        for ifreq in range(nfreqs):

            k = 2.*np.pi*freqs[ifreq]*nsub[ifreq]
            z = depths[idepth]
            r = np.sqrt( x**2 + y**2 + z**2 )
            inc[:,:,idepth,ifreq] = np.exp(1j * k * r)/(4*np.pi*r)

    return inc

def t2G(transflat,
        maxtrans,nfreqs,ndepths,inc,amps,
        fgs,nintg,du,
        subcomm):

    ncells = inc.shape[0]
    npsf = (fgs[0]['fgz'].shape[0]-ncells)//nintg
    
    trans = transflat.reshape((ncells,ncells,nfreqs,2))
    enear = np.array([],dtype=complex)
    for idepth in range(ndepths):
        for ifreq in range(nfreqs):
            tmp = (trans[:,:,ifreq,0] + 1j * trans[:,:,ifreq,1]) * inc[:,:,idepth,ifreq] * (amps[idepth,ifreq]/maxtrans[ifreq])
            enear = np.concatenate((enear,tmp.flatten()))

    enear = enear.reshape((ndepths,nfreqs, ncells,ncells))
    efar = n2f.n2f(enear, 1.,1., fgs, subcomm)
    
    G=[]
    for idepth in range(ndepths):
        for ifreq in range(nfreqs):
            G.append( (np.abs(efar[idepth,ifreq,:,:])**2).reshape((npsf,nintg,npsf,nintg)).sum(axis=(1,3)) * du**2 )
            
    return np.array(G)

test=0
if test==1:

    import params
    ncells = params.ncells
    du = params.du
    freqs = params.freqs
    depths = params.depths
    nsub = np.sqrt(params.epsSiO2)
    ngreen = params.ngreen
    Dz = params.Dz
    nintg = params.nintg
    
    inc = incident(ncells,du,freqs,depths,nsub)

    nfreqs=freqs.size
    ndepths=depths.size
    maxtrans=np.ones(nfreqs)
    amps=np.ones((ndepths,nfreqs)) * 1e5
    fgs = []
    for i in range(nfreqs):
        fgs.append( n2f.greens(ngreen,ngreen,du,du, freqs[i], 1.,1., Dz) )

    def t2ret(transflat):
        G = t2G(transflat,
                maxtrans,nfreqs,ndepths,inc,amps,
                fgs,nintg,du,
                comm)
        ret = np.sum(G**2)

        return ret
    g = grad(t2ret)
    
    np.random.seed(1234)
    
    ntot = ncells*ncells*nfreqs*2
    ndat = 50
    dp = 0.001
    tmp = np.zeros(ntot)
    for i in range(ndat):
        chk = np.random.randint(low=0,high=ntot)
        tfl = np.random.uniform(low=-3.,high=3.,size=ntot)
        gdat = g(tfl)

        tmp[:] = tfl[:]
        tmp[chk] -= dp
        obj0 = t2ret(tmp)

        tmp[:] = tfl[:]
        tmp[chk] += dp
        obj1 = t2ret(tmp)

        cendiff = (obj1-obj0)/(2.*dp)
        adj = gdat[chk]

        if comm.rank==0:
            print("check: {} {} {}".format(adj,cendiff,(adj-cendiff)/cendiff))



        
