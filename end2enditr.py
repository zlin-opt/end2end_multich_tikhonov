import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

import time
import autograd.numpy as np
import numpy as npo
from autograd import grad
import scipy.ndimage as nd
import params as prm
import iteratives as itr
import n2fdiff as n2f
import interface as itf
import nlopt
import argparse
parser=argparse.ArgumentParser()
import surrogate as sur
from julia.api import Julia
Julia(compiled_modules=False)
from julia import Main as jl
chebjacobian=jl.FastChebInterp.chebjacobian
import mpi4py.MPI as MPI
comm=MPI.COMM_WORLD
if comm.rank==0:
    print("current working directory = {}".format(cwd))
comm.Barrier()

###
def getGu(G,uflat):

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

    return Gu
    
def getGTv(G,v):

    nch = G.shape[0]
    nker = G.shape[1]
    nout = v.shape[0]
    narr = nker - nout
    
    fftv = np.fft.fft2( np.pad(v,((narr//2,narr//2),(narr//2,narr//2)),mode='constant') )
    GTv = np.zeros((0,narr,narr))
    for ich in range(nch):
        fftGflip = np.fft.fft2( itr.myflip2d(G[ich,:,:]) )
        tmp = np.real(np.array( np.fft.ifftshift( np.fft.ifft2( fftv * fftGflip ))[nout//2-1:nout//2-1+narr,nout//2-1:nout//2-1+narr], copy=True ))
        GTv = np.concatenate( (GTv,tmp.reshape((1,narr,narr))), axis=0 )

    return GTv.flatten()
###

def select_ranks():

    nodename = MPI.Get_processor_name()
    nodelist = comm.allgather(nodename)
    rnlist = comm.allgather([comm.rank,nodename])

    uniquenodes = []
    for word in nodelist:
        if word not in uniquenodes:
            uniquenodes.append(word)

    head_ranks = npo.zeros((len(uniquenodes),2),dtype=int)
    ct = 0
    for node in uniquenodes:
        for i in range(comm.size):
            if rnlist[i][1]==node:
                head_ranks[ct,0] = int(ct)
                head_ranks[ct,1] = int(rnlist[i][0])
                ct += 1
                break
                
    return head_ranks
head_ranks=select_ranks()
nnodes=head_ranks[:,0].size
if comm.rank==0:
    print("There are {} nodes with head ranks {}".format(nnodes,head_ranks[:,1]))
comm.Barrier()
#############################

parser.add_argument('--init', type=int, default=0)
parser.add_argument('--maxeval', type=int, default=10000)
parser.add_argument('--initfile', default='init.txt')
parser.add_argument('--Job', type=int, default=1)
parser.add_argument('--nsr', type=float, default=prm.nsr)
parser.add_argument('--noise_level', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--numimg', type=int, default=0)
args = parser.parse_args()

np.random.seed(args.seed)
    
freqs=prm.freqs
depths=prm.depths
ntruth=prm.ntruth
nextra=prm.nextra
npad=prm.npad
nsr=prm.nsr
tol=prm.tol
nintg=prm.nintg
du=prm.du
nsub=np.sqrt(prm.epsSiO2)
Dz=prm.Dz
cheborder = prm.order
ngeom=prm.ngeom
lb=prm.lbgeom
ub=prm.ubgeom
filename = prm.outputfile
scale=prm.scale
Job=args.Job
nfreqs=prm.nfreqs
ndepths=prm.ndepths
nch=prm.nch
nsens=prm.nsens
nshot=prm.nshot
npsf=prm.npsf
nfar=prm.nfar
ncells=prm.ncells
ngreen=prm.ngreen

imgcolor = comm.rank//nch
subcomm = comm.Split(imgcolor)
nimages = comm.size//nch
assert (comm.size%nch)==0, "choose the #cpus as a multiple of nch"

#########################################
ncells_per_proc = ncells**2//comm.size
num_rcells = (ncells**2)%comm.size
rcells = np.arange(ncells_per_proc*comm.size,ncells**2)

inc = itf.incident(ncells,du,freqs,depths,nsub)

fgs=[]
for i in range(nfreqs):
    fgs.append( n2f.greens(ngreen,ngreen,du,du, freqs[i], 1.,1., Dz) )

truths = np.zeros((nch,ntruth,ntruth,nimages))
for idepth in range(ndepths):
    for ifreq in range(nfreqs):
        ich = ifreq + nfreqs * idepth
        for img in range(nimages):
            tmp = nd.gaussian_filter(np.random.uniform(low=0.,high=1.,size=(ntruth,ntruth)),sigma=0.5)
            tmp[tmp>0.5]=1.
            tmp[tmp<=0.5]=0.
            truths[ich,:,:,img]=tmp[:,:]

model,maxtrans = sur.getmodel(cheborder,lb,ub,filename,nfreqs)
comm.Barrier()
    
### normalize the incident fields
tmp = np.zeros((ncells,ncells,nfreqs,2))
tmp[:,:,:,0]=1.
Gtmp = itf.t2G(tmp.flatten(),
               np.ones(nfreqs),
               nfreqs,ndepths,inc,
               np.ones((ndepths,nfreqs)),
               fgs,nintg,du,
               subcomm)
amps=np.zeros((ndepths,nfreqs))
for idepth in range(ndepths):
    for ifreq in range(nfreqs):
        ich = ifreq + nfreqs * idepth
        amps[idepth,ifreq] = np.sqrt( (ncells*du)**2/np.sum(Gtmp[ich]) )
        #amps[i] = np.sqrt( nsr/np.mean(Gtmp[ich]) )
        
#############
if comm.rank==0:
    print("amps for normalizing the far fields: {}".format(amps))
    print("the total number of processors is {} while the total number of cells is {}; each processor handles {} cells with {} remainder".format(comm.size,
                                                                                                                                                 ncells**2,
                                                                                                                                                 ncells_per_proc,
                                                                                                                                                 num_rcells))
    print("Total number of training points: {}".format(nimages))
comm.Barrier()
#############

def t2ret(trans):
    G = itf.t2G(trans,
                maxtrans,nfreqs,ndepths,inc,amps,
                fgs,nintg,du,
                subcomm)
    u = truths[:,:,:,imgcolor].flatten()
    uest = itr.convtikh(G,u,
                        nsr,tol,
                        subcomm)
    ret = np.linalg.norm(u-uest)/np.linalg.norm(u)
    return ret
t2ret_grad = grad(t2ret)

count = [0]
def dof2ret(dof,gdat):

    if comm.rank==0:
        np.savetxt('dof{}_MMA_nsr{}_init{}_seed{}.txt'.format(count[0],nsr,args.init,args.seed),dof)
    comm.Barrier()

    tmpdof = np.zeros(dof.size)
    tmpdof[:] = dof[:]
    geoms = tmpdof.reshape((ncells**2,ngeom))

    t0 = time.time()
    trans_loc = np.zeros((ncells**2,nfreqs*2))
    jac_loc = np.zeros((ncells**2,nfreqs*2,ngeom))
    for i in range(ncells_per_proc):
        icell = i + ncells_per_proc * comm.rank
        tmp = chebjacobian(model,geoms[icell,:])
        trans_loc[icell,:] = np.array(tmp[0])
        jac_loc[icell,:,:] = np.array(tmp[1])
    if comm.rank < num_rcells:
        icell = rcells[comm.rank]
        tmp = chebjacobian(model,geoms[icell,:])
        trans_loc[icell,:] = np.array(tmp[0])
        jac_loc[icell,:,:] = np.array(tmp[1])
    comm.Barrier()
    trans = np.empty((ncells*ncells,nfreqs*2))
    jac = np.empty((ncells*ncells,nfreqs*2,ngeom))
    comm.Allreduce(trans_loc, trans, op=MPI.SUM)
    comm.Allreduce(jac_loc, jac, op=MPI.SUM)
    trans = trans.flatten()
    t1 = time.time()
    if comm.rank==0:
        print("the physics model takes {} sec".format(t1-t0))
    comm.Barrier()

    t0 = time.time()
    ret = t2ret(trans)
    comm.Barrier()
    ret = comm.allreduce(ret, op=MPI.SUM)
    ret = ret/(nch*nimages)
    t1 = time.time()
    if comm.rank==0:
        print("image processing takes {} sec".format(t1-t0))
        print("mse at step {} is {}".format(count[0],ret))
    comm.Barrier()

    if gdat.size > 0:
        t0 = time.time()
        gvec_loc = np.reshape(t2ret_grad(trans),(ncells**2,nfreqs*2))
        gvec = np.empty((ncells**2,nfreqs*2))
        comm.Allreduce(gvec_loc, gvec, op=MPI.SUM)
        t1 = time.time()
        if comm.rank==0:
            print("backprop through image processing takes {} sec".format(t1-t0))
        comm.Barrier()
        
        t0 = time.time()
        tmp = np.zeros((ncells**2,ngeom))
        for i in range(ncells_per_proc):
            icell = i + ncells_per_proc * comm.rank
            tmp[icell,:] = np.dot(gvec[icell,:],jac[icell,:,:])
        if comm.rank<num_rcells:
            icell = rcells[comm.rank]
            tmp[icell,:] = np.dot(gvec[icell,:],jac[icell,:,:])
        adj = np.empty((ncells**2,ngeom))
        comm.Allreduce(tmp, adj, op=MPI.SUM)
        adj = adj.flatten()
        t1 = time.time()
        if comm.rank==0:
            print("final vjp takes {} sec".format(t1-t0))
        comm.Barrier()
        
        gdat[:] = adj[:]/(nch*nimages*scale)
        if comm.rank==0:
            print("Max abs(grad) at step {} is {}".format(count[0],np.max(np.abs(gdat))))
            print("Min abs(grad) at step {} is {}".format(count[0],np.min(np.abs(gdat))))
            print("Mean abs(grad) at step {} is {}".format(count[0],np.mean(np.abs(gdat))))
        comm.Barrier()

    count[0] = count[0] + 1

    return ret/scale

if Job==0:
    n=ncells*ncells*ngeom
    ndat = 100
    dp = 0.000001
    tmp = np.zeros(n)
    for i in range(ndat):

        dof = np.zeros((ncells**2,ngeom))
        for igeom in range(ngeom):
            dof[:,igeom]=np.random.uniform(low=lb[igeom],high=ub[igeom],size=ncells**2)
        dof = dof.flatten()
        gdat = np.zeros(dof.size)
        dof2ret(dof,gdat)
        
        chk = np.random.randint(low=0,high=n)
        tmp[:] = dof[:]
        tmp[chk] -= dp
        obj0 = dof2ret(tmp,np.array([]))

        tmp[:] = dof[:]
        tmp[chk] += dp
        obj1 = dof2ret(tmp,np.array([]))

        cendiff = (obj1-obj0)/(2.*dp)
        adj = gdat[chk]
        
        if comm.rank==0:
            print("check: {} {} {}".format(adj,cendiff,(adj-cendiff)/adj))
    

if Job==1:

    maxeval=args.maxeval
    init=args.init
    filename=args.initfile
            
    if comm.rank==0:
        print('Optimization Job [1] is chosen.')
        print("Job 1 maxeval: {}".format(maxeval))
        print("Job 1 init: {}".format(init))
        print("Job 1 initial file: {}".format(filename))
    comm.Barrier()

    if init==0:
        dof0=np.zeros((ncells**2,ngeom))
        for igeom in range(ngeom):
            dof0[:,igeom]= np.ones(ncells**2) * (lb[igeom]+ub[igeom])/2.
        dof0 = dof0.flatten()
    else:
        dof0 = np.loadtxt(filename)

    lbopt = np.zeros((ncells**2,ngeom))
    ubopt = np.zeros((ncells**2,ngeom))
    for igeom in range(ngeom):
        lbopt[:,igeom] = lb[igeom] * np.ones(ncells**2)
        ubopt[:,igeom] = ub[igeom] * np.ones(ncells**2)
    lbopt = lbopt.flatten()
    ubopt = ubopt.flatten()

    opt = nlopt.opt(nlopt.LD_MMA, dof0.size)
    opt.set_min_objective(dof2ret)
    opt.set_lower_bounds(lbopt)
    opt.set_upper_bounds(ubopt)
    opt.set_ftol_rel(1e-16)
    opt.set_maxeval(maxeval)
    if comm.rank==0:
        opt.set_param("verbosity", 1)
    comm.Barrier()
    xopt = opt.optimize(dof0)

    print('Optimization returns {}'.format(xopt))
    print('Optimization Done!')

    

if Job==2:

    import h5py as hp
    
    optfile=args.initfile
    
    dof = np.loadtxt(optfile)
    dof = dof.reshape((ncells**2,ngeom))

    t0 = time.time()
    trans_loc = np.zeros((ncells**2,nfreqs*2))
    for i in range(ncells_per_proc):
        icell = i + ncells_per_proc * comm.rank
        tmp = chebjacobian(model,dof[icell,:])
        trans_loc[icell,:] = np.array(tmp[0])
    if comm.rank < num_rcells:
        icell = rcells[comm.rank]
        tmp = chebjacobian(model,dof[icell,:])
        trans_loc[icell,:] = np.array(tmp[0])
    comm.Barrier()
    trans = np.empty((ncells*ncells,nfreqs*2))
    comm.Allreduce(trans_loc, trans, op=MPI.SUM)
    trans = trans.flatten()
    t1 = time.time()
    if comm.rank==0:
        print("the physics model takes {} sec".format(t1-t0))
    comm.Barrier()

    G = itf.t2G(trans,
                maxtrans,nfreqs,ndepths,inc,amps,
                fgs,nintg,du,
                subcomm)

    u = np.zeros((nch,ntruth,ntruth))
    if args.numimg==1:
        fid = hp.File('pics.h5','r')
        for ich in range(nch):
            tmp = np.array( fid['pic{}'.format(ich)] )
            u[ich,:,:]=tmp[:,:]
        fid.close()
    else:
        np.random.seed(args.seed*(imgcolor+1))
        for idepth in range(ndepths):
            for ifreq in range(nfreqs):
                ich = ifreq + nfreqs * idepth
                tmp = nd.gaussian_filter(np.random.uniform(low=0.,high=1.,size=(ntruth,ntruth)),sigma=1.0)
                tmp[tmp>0.5]=1.
                tmp[tmp<=0.5]=0.
                u[ich,:,:]=tmp[:,:]        
    uflat = u.flatten()

    Gu = getGu(G,uflat)
    noise_sigma = args.noise_level * np.mean(Gu)
    eta = np.random.normal(loc=0.,scale=noise_sigma,size=Gu.shape)
    v = Gu + np.abs(eta)
    GTv = getGTv(G,v)

    t0 = time.time()
    uest,res,itr = itr.cga(G,GTv, itr.GTGu_fftG, args.nsr,tol, subcomm)
    t1 = time.time()
    if subcomm.rank==0:
        print("forward cga converges in {} steps with relative residual {}, taking {} sec".format(itr,res,t1-t0))
                            
    ret = np.linalg.norm(uflat-uest)/np.linalg.norm(uflat)
    if subcomm.rank==0:
        print("mse is {} for ground truth {}; nsr is {}; noise-level is {}".format(ret,imgcolor,args.nsr,args.noise_level))
    comm.Barrier()

    if comm.rank==0:
        u = uflat.reshape((nch,ntruth,ntruth))
        uest = uest.reshape((nch,ntruth,ntruth))
        fid = hp.File('visual_{}_noiselevel{}_reconnsr{}.h5'.format(optfile[:-4],args.noise_level,args.nsr),'w')
        dof = dof.flatten().reshape((ncells,ncells,ngeom))
        for igeom in range(ngeom):
            fid.create_dataset('geom{}'.format(igeom),data=dof[:,:,igeom])
        for ich in range(nch):
            fid.create_dataset('G{}'.format(ich),data=G[ich,:,:])
            fid.create_dataset('sqrtG{}'.format(ich),data=np.sqrt(np.abs(G[ich,:,:])))
            data = np.concatenate((u[ich,:,:],-1.*np.ones((ntruth,1)),uest[ich,:,:]),axis=1)
            fid.create_dataset('img{}'.format(ich),data=data)
        fid.create_dataset('vorig',data=Gu)
        fid.create_dataset('vnoisy',data=v)
        fid.close()

    comm.Barrier()

    
if Job==3:

    maxeval=args.maxeval
    init=args.init
    filename=args.initfile
            
    if comm.rank==0:
        print('ADAM Optimization Job [3] is chosen.')
        print("Job 1 maxeval: {}".format(maxeval))
        print("Job 1 init: {}".format(init))
        print("Job 1 initial file: {}".format(filename))
    comm.Barrier()

    if init==0:
        dof=np.zeros((ncells**2,ngeom))
        for igeom in range(ngeom):
            dof[:,igeom]= np.ones(ncells**2) * (lb[igeom]+ub[igeom])/2.
        dof = dof.flatten()
    else:
        dof = np.loadtxt(filename)
    gdat = np.zeros(dof.size)
        
    lbopt = np.zeros((ncells**2,ngeom))
    ubopt = np.zeros((ncells**2,ngeom))
    for igeom in range(ngeom):
        lbopt[:,igeom] = lb[igeom] * np.ones(ncells**2)
        ubopt[:,igeom] = ub[igeom] * np.ones(ncells**2)
    lbopt = lbopt.flatten()
    ubopt = ubopt.flatten()

    ######
    alpha = args.alpha
    beta_1 = 0.9
    beta_2 = 0.999  #initialize the values of the parameters
    epsilon = 1e-8


    m_t = np.zeros(dof.size)
    v_t = np.zeros(dof.size)
    for ieval in range(1,maxeval):

        truths = np.random.uniform(low=0.,high=1.,size=(nch,ntruth,ntruth,nimages))
        ret = dof2ret(dof,gdat)

        m_t = beta_1*m_t + (1-beta_1)*gdat   #updates the moving averages of the gradient
        v_t = beta_2*v_t + (1-beta_2)*(gdat*gdat)    #updates the moving averages of the squared gradient
        m_cap = m_t/(1-(beta_1**ieval))   #calculates the bias-corrected estimates
        v_cap = v_t/(1-(beta_2**ieval))   #calculates the bias-corrected estimates

        dof = dof - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)  #updates the parameters

        dof[dof<lbopt]=lb[0]
        dof[dof>ubopt]=ub[0]

    comm.Barrier()

