import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import autograd
import mpi4py.MPI as MPI
comm=MPI.COMM_WORLD
from julia.api import Julia
Julia(compiled_modules=False)
from julia import Main as jl
jl.include("FastChebInterp.jl")
chebfit = jl.FastChebInterp.chebfitv1
chebjacobian=jl.FastChebInterp.chebjacobian

def getmodel(n,lb,ub,filename,nfreqs):

    raw = np.loadtxt(filename)
    vals = np.zeros((nfreqs*2, n+1))
    for i in range(nfreqs):
        vals[0+2*i,:]=raw[:,4+0+2*i].reshape((n+1))
        vals[1+2*i,:]=raw[:,4+1+2*i].reshape((n+1))

    model = chebfit(vals, lb, ub)

    return model,np.ones(nfreqs)

validate=0
if validate==1:

    import rcwa
    import params

    lb=params.lbgeom
    ub=params.ubgeom
    order=params.order
    filename=params.outputfile
    nfreqs=params.nfreqs
    model,_ = getmodel(order,lb,ub,filename,nfreqs)

    np.random.seed(1234)
    nsamples = 100
    freqs = params.freqs
    epsdevice = params.epsTiO2
    epssubstrate = params.epsSiO2
    samples = np.random.uniform(low=lb[0],high=ub[0],size=nsamples)
    ifreq = np.random.randint(low=0,high=nfreqs,size=nsamples)
    data = np.zeros((nsamples,7))
    for i in range(nsamples):
        f,g = chebjacobian(model, [samples[i]])
        f = np.array(f).reshape((nfreqs,2))
        ex = rcwa.getfields(samples[i],
                            freqs[ifreq[i]],
                            epsdevice[ifreq[i]],epssubstrate[ifreq[i]],
                            nG=500)
        sur = f[ifreq[i],0] + 1j * f[ifreq[i],1]
        err0 = np.real(ex-sur)/np.real(ex)
        err1 = np.imag(ex-sur)/np.imag(ex)
        data[i,0] = samples[i]
        data[i,1] = err0
        data[i,2] = err1
        data[i,3] = np.real(sur)
        data[i,4] = np.real(ex)
        data[i,5] = np.imag(sur)
        data[i,6] = np.imag(ex)
        print("sample {}, error: {} {}, sur,ex real: {} {}, sur,ex imag: {} {}".format( data[i,0], data[i,1],data[i,2], data[i,3],data[i,4], data[i,5],data[i,6] ))
    np.savetxt('validate.dat',data)
    print("mean error real, imag: {} {}".format( np.mean(np.abs(data[:,1])), np.mean(np.abs(data[:,2])) ))

view=0
if view==1:

    import params
    
    lb=params.lbgeom
    ub=params.ubgeom
    order=params.order
    filename=params.outputfile
    nfreqs=params.nfreqs
    model,_ = getmodel(order,lb,ub,filename,nfreqs)

    np.random.seed(1234)
    nsamples = 2000
    freqs = params.freqs
    epsdevice = params.epsTiO2
    epssubstrate = params.epsSiO2
    samples = np.linspace(lb[0],ub[0],nsamples) 
    ifreq = 0*np.ones(nsamples,dtype=int) 
    data = np.zeros((nsamples,3))
    for i in range(nsamples):
        f,g = chebjacobian(model, [samples[i]])
        f = np.array(f).reshape((nfreqs,2))
        sur = f[ifreq[i],0] + 1j * f[ifreq[i],1]
        data[i,0] = samples[i]
        data[i,1] = np.real(sur)
        data[i,2] = np.imag(sur)
    np.savetxt('viewsample.dat',data)

