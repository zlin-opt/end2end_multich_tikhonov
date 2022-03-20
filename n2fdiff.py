import autograd.numpy as np
from autograd import grad
import numpy.fft as fft
from autograd.extend import primitive, defvjp
import mpi4py.MPI as MPI
comm=MPI.COMM_WORLD
import time

def greens(nx,ny,dx,dy, freq, eps,mu, Dz):

    omega = 2.*np.pi*freq
    n = np.sqrt(eps*mu)
    k = n*omega
    
    Lx, Ly = nx*dx, ny*dy

    x, y= np.meshgrid( np.linspace(-Lx/2.,Lx/2.-dx,nx),
                       np.linspace(-Ly/2.,Ly/2.-dy,ny) )

    r = np.sqrt( x**2 + y**2 + Dz**2 )
    expfac = np.exp(1j * k * r)
    dxy = dx*dy
    
    g = expfac/(4*np.pi*r) *dxy
    gz = Dz * (-1. + 1j * k * r) * expfac/(4.*np.pi*r**3) * dxy
    
    fgz  = fft.fft2(gz)
    fgzT = fft.fft2(np.flip(gz))
    
    ret = {"fgz" : fgz, "fgzT" : fgzT}

    return ret

def fftconv2d(arr,fftker,fwd=1):

    narr = arr.shape[0]
    nker = fftker.shape[0]
    nout = nker - narr

    fftarr = fft.fft2( np.pad(arr,((nout//2,nout//2),(nout//2,nout//2)),mode='constant') )
    if fwd==1:
        out = np.array( fft.ifftshift( fft.ifft2( fftarr * fftker ))[narr//2:narr//2+nout,narr//2:narr//2+nout], copy=True )
    else:
        out = np.array( fft.ifftshift( fft.ifft2( fftarr * fftker ))[-1+narr//2:-1+narr//2+nout,-1+narr//2:-1+narr//2+nout], copy=True )
    
    return out

#enear[depths,freqs, nx,ny] -- complex-valued
@primitive
def n2f(enear, eps,mu, fgs, subcomm):
    
    ndepths=enear.shape[0]
    nfreqs=enear.shape[1]

    nnear = enear.shape[2]
    ngreen = fgs[0]["fgz"].shape[0]
    nfar = ngreen-nnear

    efar_all = np.empty(ndepths*nfreqs*nfar*nfar,dtype=complex)

    t0=time.time()
        
    idepth,ifreq = np.unravel_index(subcomm.rank,(ndepths,nfreqs))
    arr = enear[idepth,ifreq, :,:] 
    fgz = fgs[ifreq]["fgz"]
    efar = fftconv2d(arr,fgz,fwd=1) * (-mu/eps)
    subcomm.Allgather(efar.flatten(),efar_all)

    t1=time.time()
    if subcomm.rank==0:
        print(">>> mpi convolution and Allgather in forward direction takes {} sec".format(t1-t0))
    comm.Barrier()

    return efar_all.reshape((ndepths,nfreqs, nfar,nfar))

def f2n(vec, eps,mu, fgs, subcomm):

    ndepths=vec.shape[0]
    nfreqs=vec.shape[1]

    nfar = vec.shape[2]
    ngreen = fgs[0]["fgz"].shape[0]
    nnear = ngreen-nfar

    vjp_all = np.empty(ndepths*nfreqs*nnear*nnear,dtype=complex)

    t0=time.time()
    
    idepth,ifreq = np.unravel_index(subcomm.rank,(ndepths,nfreqs))
    arr = vec[idepth,ifreq, :,:] 
    fgzT = fgs[ifreq]["fgzT"]
    vjp = fftconv2d(arr,fgzT,fwd=0) * (-mu/eps)
    subcomm.Allgather(vjp.flatten(),vjp_all)

    t1=time.time()
    if subcomm.rank==0:
        print("<<< mpi convolution and Allgather in adjoint direction takes {} sec".format(t1-t0))
    comm.Barrier()

    return vjp_all.reshape((ndepths,nfreqs, nnear,nnear))

def grad_n2f(ans, enear, eps,mu, fgs, subcomm):

    def vecjacprod(vec):
        return f2n(vec, eps,mu, fgs, subcomm)

    return vecjacprod

defvjp(n2f,
       grad_n2f,
       argnums=[0])

################
test=0
if test==1:
    depths=np.array([1e3,1e5])
    freqs=np.array([0.8,1.0,1.2])
    nnear=2000
    nfar=2000
    du=0.9
    Dz=1e4

    ndepths=depths.size
    nfreqs=freqs.size
    ngreen=nnear+nfar

    t0=time.time()
    fgs=[]
    for ifreq in range(nfreqs):
        fgs.append(greens(ngreen,ngreen,du,du, freqs[ifreq],1.,1., Dz))
    comm.Barrier()
    t1=time.time()
    if comm.rank==0:
        print("building fgs takes {} sec".format(t1-t0))
    
    def test(dof):

        enear = np.reshape(dof[:dof.size//2] + 1j * dof[dof.size//2:],(ndepths,nfreqs,nnear,nnear))
        efar = n2f(enear, 1.,1., fgs, comm)
        ret = np.sum(np.real(efar)*np.imag(efar))
        
        return ret
    gfun=grad(test)

    np.random.seed(5678)
    ntot = 2*ndepths*nfreqs*nnear*nnear
    ndat = 100
    dp = 0.0001
    tmp = np.zeros(ntot)
    for i in range(ndat):
        chk = np.random.randint(low=0,high=ntot)
        dof = np.random.uniform(low=-3.,high=3.,size=ntot)
        gdat = gfun(dof)

        tmp[:] = dof[:]
        tmp[chk] -= dp
        obj0 = test(tmp)

        tmp[:] = dof[:]
        tmp[chk] += dp
        obj1 = test(tmp)

        cendiff = (obj1-obj0)/(2*dp)
        adj = gdat[chk]

        if comm.rank==0:
            print("check: {} {} {} {}".format(i,adj,cendiff,(adj-cendiff)/cendiff))

    comm.Barrier()
    
