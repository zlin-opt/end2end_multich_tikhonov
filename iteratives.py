import autograd.numpy as np
from autograd import grad
from autograd.extend import primitive, defvjp
import time
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD

def cga(A,b,
        matvec,l,
        tol,
        subcomm,
        uorig = np.array([])):

    nch = A.shape[0]
    fftA = np.zeros(A.shape,dtype=complex)
    fftAflip = np.zeros(A.shape,dtype=complex)
    for ich in range(nch):
        fftA[ich,:,:] = np.fft.fft2(A[ich,:,:])
        fftAflip[ich,:,:] = np.fft.fft2(np.flip(A[ich,:,:]))

    if uorig.size > 0:
        x = np.array(uorig,copy=True)
    else:
        x = np.zeros(b.shape)

    r = b - ( matvec(fftA,fftAflip,x,subcomm) + l*x )
    p = np.array(r, copy=True)

    rsold = np.sum(r*r)

    for i in range(b.flatten().size):

        Ap = matvec(fftA,fftAflip,p,subcomm) + l*p

        alpha = rsold/np.sum(p*Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = np.sum(r*r)

        if np.sqrt(rsnew)<tol:
            break
        p = r + (rsnew/rsold) * p

        rsold = 1. * rsnew

    return x,rsnew,i

########
def myflip2d(x):

    m,n = x.shape[0],x.shape[1]
    
    tmp1 = np.zeros((0,n))
    for i in range(m):
        tmp1 = np.concatenate((tmp1,x[m-i-1,:].reshape((1,n))),axis=0)
        
    tmp2 = np.zeros((m,0))
    for i in range(n):
        tmp2 = np.concatenate((tmp2,tmp1[:,n-i-1].reshape((m,1))),axis=1)
        
    return tmp2


def GTGu(G,uflat):

    nch = G.shape[0]
    nker= G.shape[1]
    narr = int(np.sqrt(uflat.size//nch))
    nout = nker - narr

    u = uflat.reshape((nch,narr,narr))
    
    fftGu = np.zeros((nker,nker),dtype=complex)
    for ich in range(nch):
        fftG = np.fft.fft2( G[ich,:,:] )
        fftarr = np.fft.fft2( np.pad(u[ich,:,:],((nout//2,nout//2),(nout//2,nout//2)),mode='constant') )
        fftGu += fftarr * fftG
    Gu = np.real(np.array( np.fft.ifftshift( np.fft.ifft2( fftGu ))[narr//2:narr//2+nout,narr//2:narr//2+nout],copy=True))

    fftGu = np.fft.fft2( np.pad(Gu,((narr//2,narr//2),(narr//2,narr//2)),mode='constant') )
    GTGu = np.zeros((0,narr,narr))
    for ich in range(nch):
        fftGflip = np.fft.fft2( myflip2d(G[ich,:,:]) )
        tmp = np.real(np.array( np.fft.ifftshift( np.fft.ifft2( fftGu * fftGflip ))[nout//2-1:nout//2-1+narr,nout//2-1:nout//2-1+narr], copy=True ))
        GTGu = np.concatenate( (GTGu,tmp.reshape((1,narr,narr))), axis=0 )

    return GTGu.flatten()

def GTGu_fftG(fftG,fftGflip,uflat,
              subcomm):

    ich = subcomm.rank
    
    nch = fftG.shape[0]
    nker=fftG.shape[1]
    narr = int(np.sqrt(uflat.size//nch))
    nout = nker - narr

    u = uflat.reshape((nch,narr,narr))

    fftGu = np.empty((nker,nker),dtype=complex)
    fftarr = np.fft.fft2( np.pad(u[ich,:,:],((nout//2,nout//2),(nout//2,nout//2)),mode='constant') )
    subcomm.Allreduce(fftarr*fftG[ich,:,:],fftGu,op=MPI.SUM)
    Gu = np.real(np.array( np.fft.ifftshift( np.fft.ifft2( fftGu ))[narr//2:narr//2+nout,narr//2:narr//2+nout],copy=True))

    fftGu = np.fft.fft2( np.pad(Gu,((narr//2,narr//2),(narr//2,narr//2)),mode='constant') )
    GTGu = np.empty((nch,narr*narr))
    local = np.real(np.array( np.fft.ifftshift( np.fft.ifft2( fftGu * fftGflip[ich,:,:] ))[nout//2-1:nout//2-1+narr,nout//2-1:nout//2-1+narr], copy=True ))
    subcomm.Allgather(local.flatten(),GTGu)
        
    return np.array(GTGu).flatten()


def vDu(G,v,u):

    nch = G.shape[0]
    nG = G.shape[1]

    def vGTGu(Gflat):
        Gfull = Gflat.reshape((nch,nG,nG))
        return np.sum( v * GTGu(Gfull,u) )

    g = grad(vGTGu)

    ret = vGTGu(G.flatten())
    gdat = g(G.flatten()).reshape((nch,nG,nG))
    return ret, gdat
        
#####
# solves uest for (GTG + l I) uest = GTGu
# computes v * duest/dG
@primitive
def convtikh(G, u, l, tol, subcomm):

    b = GTGu(G,u)
    t0 = time.time()
    uest,res,itr = cga(G,b, GTGu_fftG, l,tol, subcomm)
    t1 = time.time()
    if subcomm.rank==0:
        print("forward cga converges in {} steps with relative residual {}, taking {} sec".format(itr,res,t1-t0))
    
    return uest

def grad_convtikh(ans, G, u, l, tol, subcomm):

    def vecjacprod(vec):

        adjvar,res,itr = cga(G,vec, GTGu_fftG, l,tol, subcomm)
        t0 = time.time()
        foo,gdat = vDu(G,adjvar,u-ans) 
        t1 = time.time()
        if subcomm.rank==0:
            print("adjoint cga converges in {} steps with relative residual {}, taking {} sec".format(itr,res,t1-t0))
        
        return gdat

    return vecjacprod

defvjp(convtikh,
       grad_convtikh,
       argnums=[0])

########
test=0
if test==1:

    np.random.seed(1234)

    nch = 4
    imgcolor = comm.rank//nch
    subcomm = comm.Split(imgcolor)
    nimgs = comm.size//nch
    
    n = 30
    m = int(n*int(np.ceil(np.sqrt(nch)))) + 10
    if m%2 == 1:
        m = m + 1
    u = np.random.uniform(low=0.,high=1.,size=(nimgs,nch*n*n))
    l = 100.0
    tol = 1e-16
    
    def uest2sum(Gflat):
        G = Gflat.reshape((nch,m+n,m+n))
        uest = convtikh(G,u[imgcolor,:],l,tol,subcomm)
        ret = np.linalg.norm(u[imgcolor,:]-uest)/np.linalg.norm(u[imgcolor,:])
        return ret
        
    g = grad(uest2sum)

    ndat = 100
    dp = 0.001
    ntot = nch*(m+n)**2
    tmp = np.zeros(ntot)

    for i in range(ndat):
        chk = np.random.randint(low=0,high=ntot)
        Gflat = np.random.uniform(low=0.,high=1.,size=ntot)
        gdat = g(Gflat)
        adj = gdat[chk]
        
        tmp[:] = Gflat[:]
        tmp[chk] -= dp
        obj0 = uest2sum(tmp)

        tmp[:] = Gflat[:]
        tmp[chk] += dp
        obj1 = uest2sum(tmp)

        cendiff = (obj1-obj0)/(2.*dp)

        if subcomm.rank==0:
            print("imgID{}, check: {} {} {}".format(imgcolor,adj,cendiff,(adj-cendiff)/cendiff))
        
