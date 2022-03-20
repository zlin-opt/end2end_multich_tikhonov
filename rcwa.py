import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import grcwa as rc
import autograd.numpy as np
import params

Lx=params.du
Ly=params.du
thickness=params.thickness/params.wavcen

L1 = [Lx,0]
L2 = [0,Ly]
Nx = 10000
Ny = 10000

def getfields(side,
              freq,
              epsdevice,epssubstrate,
              nG=500):
    
    obj = rc.obj(int(nG),L1,L2,freq,0.,0.,verbose=1)

    obj.Add_LayerUniform(1.0,1.0)
    obj.Add_LayerUniform(1000.,epssubstrate)
    obj.Add_LayerGrid(thickness,Nx,Ny)
    obj.Add_LayerUniform(1.0,1.0)
    obj.Add_LayerUniform(1.0,1.0)

    obj.Init_Setup()

    obj.a0 = np.zeros(2*obj.nG,dtype=complex)
    obj.a0[0] = 1.+0.*1j
    obj.bN = np.zeros(2*obj.nG,dtype=complex)
    
    epgrid = np.zeros((Nx,Ny))
    xarr = np.linspace(-Lx/2.,Lx/2.,Nx)
    yarr = np.linspace(-Ly/2.,Ly/2.,Ny)
    x, y = np.meshgrid(xarr,yarr,indexing='ij')
    ind = np.logical_and(abs(x) < side/2., abs(y) < side/2.)
    epgrid[ind]=1.0

    epgrid = (epsdevice-1.) * epgrid + 1.
    obj.GridLayer_geteps(epgrid.flatten())
    
    aN,b0 = obj.GetAmplitudes(4,0.1)
    ret = aN[0]/obj.a0[0]

    return ret

test=0
if test==1:

    from julia.api import Julia
    Julia(compiled_modules=False)
    from julia import Main as jl
    jl.include("FastChebInterp.jl")
    chebfit = jl.FastChebInterp.chebfitv1
    chebpts = jl.FastChebInterp.chebpoints
    chebjacobian=jl.FastChebInterp.chebjacobian
    
    nG = 50
    ifreq = 0
    lb=params.lbgeom
    ub=params.ubgeom
    ngeom=params.ngeom
    order = []
    for i in range(ngeom):
        order.append(1279)
            
    points = np.array(chebpts(order,lb,ub))
    points = points.reshape((int((order[0]+1)**ngeom),ngeom))

    gendat=1
    if gendat==1:
        npts = points.shape[0]
        data = np.zeros((npts,3))
        for i in range(points.shape[0]):
            side = points[i,0]
            freq = params.freqs[ifreq]
            epsD = params.epsTiO2[ifreq]
            epsS = params.epsSiO2[ifreq]
            ex = getfields(side,
                           freq,
                           epsD,epsS,
                           nG=nG)
            data[i,0]=side
            data[i,1]=np.real(ex)
            data[i,2]=np.imag(ex)
            print(data[i,:])

        np.savetxt('test_rcwa_nG{}_ifreq{}_cheb{}.dat'.format(nG,ifreq,order[0]),data)

    data = np.loadtxt('test_rcwa_nG{}_ifreq{}_cheb{}.dat'.format(nG,ifreq,order[0]))

    val = np.zeros((2,order[0]+1))
    val[0,:]=data[:,1]
    val[1,:]=data[:,2]
    model = chebfit(val,lb,ub)

    gensdat=0
    nsamples=2000
    samples=np.linspace(lb[0],ub[0],nsamples)
    sdat = np.zeros((nsamples,5))
    for i in range(nsamples):
        f,g = chebjacobian(model, [samples[i]] )
        sdat[i,0] = samples[i]
        sdat[i,1] = np.array(f)[0]
        sdat[i,2] = np.array(f)[1]

        if gensdat==1:
            side = samples[i]
            freq = params.freqs[ifreq]
            epsD = params.epsTiO2[ifreq]
            epsS = params.epsSiO2[ifreq]
            ex = getfields(side,
                           freq,
                           epsD,epsS,
                           nG=nG)
            sdat[i,3] = np.real(ex)
            sdat[i,4] = np.imag(ex)
        
    np.savetxt('partialtest_suronly_nG{}_ifreq{}_cheb{}.dat'.format(nG,ifreq,order[0]),sdat)
