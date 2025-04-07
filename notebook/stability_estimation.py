from ngsolve import *
from netgen.occ import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import dualMeshInnerProducts as dmIP

def getLowestPolynomialOrder(dim, diffform, swapHDHC):
    '''returns the polynomial order of the Whitneyforms.
    dim needs to be 2 or 3, difform determines the integer l for the l-form, difform <= dim,
    swapHDHC is a bool, when false: in 2D 1-forms are H(div) elements, in 3D 1-forms are H(curl) and 2-forms H(div) elements,
    when True H(div) is replaced by H(curl) and vice versa.'''
    match dim:
        case 2:
            if swapHDHC == False:
                orders = [1,0,0]
            else:
                orders = [1,1,0]
        case 3:
            if swapHDHC == False:
                orders = [1,1,0,0]
            else:
                orders = [1,0,1,0]

    return orders[diffform]

def getFunctionSpace(dim, diffform, swapHDHC):
    match dim:
        case 2:
            if swapHDHC == False:
                differentialforms = [H1, HDiv, L2]
            else:
                differentialforms = [H1, HCurl, L2]
        case 3:
            if swapHDHC == False:
                differentialforms = [H1, HCurl, HDiv, L2]
            else:
                differentialforms = [H1, HDiv, HCurl, L2]
    
    return differentialforms[diffform]

def getInnerProduct(dim, difform, swapHDHC):
    match difform:
        case 0:
            return lambda a,b: a * b + grad(a) * grad(b)
        case 1:
            if (dim == 2 and swapHDHC == False) or (dim == 3 and swapHDHC == True):
                return lambda a,b: a * b + div(a) * div(b)
            elif (dim == 2 and swapHDHC == True) or (dim == 3 and swapHDHC == False):
                return lambda a,b: a * b + curl(a) * curl(b)
        case 2:
            if dim != 2:
                if swapHDHC == False:
                    return lambda a,b: a * b + div(a) * div(b)
                else:
                    return lambda a,b: a * b + curl(a) * curl(b)
            else:
                return lambda a,b: a * b 
        case 3:
            return lambda a,b: a * b 

def createMesh(mw, numRef, mt, b, d):
    offset = 0
    if b != '':
        offset = 1

    match d:
        case 2:
            domain = unit_square
        case 3:
            domain = unit_cube

    match mt:
        case "str":
            netgen_mesh = domain.GenerateMesh(maxh=1)
            mesh=Mesh(netgen_mesh)
            for i in range(numRef + offset):
                mesh.Refine()
        case _:
            netgen_mesh = domain.GenerateMesh(maxh=mw, segmentsperedge=(numRef + offset)+1.2, grading=0.1)
            mesh=Mesh(netgen_mesh)
    
    return mesh

def estimate_stability(num_mw = 4, bnd = '', diff_form = 0,dim = 2, mesh_type = "str", num_HO = 3, Swap_HCurl_HDiv = False):
    #matrix vector products for scipy Eigenvalue solver     
    def matvec_b(v):
        tmp1.data = v
        tmp2.data = b * tmp1
        return tmp2.FV().NumPy()

    def matvec_c(v):
        tmp3.data = v
        tmp4.data = c.mat * tmp3
        return tmp4.FV().NumPy()
        
    #l is number of meshwidths, the n-th meshwidth is 1/(2^(n-1))
    l = num_mw
    meshwidths = np.ones(l) 
    for h in range(l-1):
        meshwidths[h+1] = meshwidths[h]/2
    
    if bnd != '':
        #when there are bnd conditions, start at 0.5 meshwidth to ensure there are interior dofs 
        meshwidths *= 0.5


    print(meshwidths)
    #order for which the elements corresponds to the whitney forms
    low_order = getLowestPolynomialOrder(dim, diff_form, Swap_HCurl_HDiv)

    #amount of orders for the high-Order space
    high_orders = num_HO 
    #array storing the minimal Eigenvalue for given orders and meshwidth
    minEV = np.zeros((high_orders,l))

    fes = getFunctionSpace(dim,diff_form,Swap_HCurl_HDiv)

    InnerProduct = getInnerProduct(dim,diff_form,Swap_HCurl_HDiv)

    #high order needs to be greater than low order
    lowest_high_Order = low_order + 1
    #largest high order to test
    highest_high_order = lowest_high_Order + high_orders
    for i in range(lowest_high_Order, highest_high_order):
        print("polynomal order of low order space:", low_order, ", polynomal order of high order space:", i)
        for k in range(l):
            mesh = createMesh(mw = meshwidths[k], numRef=k, mt=mesh_type, b=bnd, d=dim)

            print("h=",meshwidths[k], "\n")

            #need to compress to remove DOFs of unrefined mesh after refinement
            H_h = Compress(fes(mesh, order = low_order, dirichlet=bnd, RT=True, type1=True)) # main function space
            w_h = GridFunction(H_h) #define vector for matvec wrapper
            H_H = Compress(fes(mesh, order = i, dirichlet=bnd, RT=True, type1=True)) # high order Function space for Riesz representative  

            print("# DoFs of low order space:", H_h.ndof, ", # DoFs of high order space:", H_H.ndof)

            #low order
            u_h = H_h.TrialFunction() # define the trial function
            v_h = H_h.TestFunction() # define the test function
            #High order
            u_H = H_H.TrialFunction() # define the trial function
            v_H= H_H.TestFunction() # define the test function

            m = BilinearForm(H_h, symmetric=True, symmetric_storage=True) #define m with the L2 inner product on the low order space
            m += u_h*v_h * dx 
            m.Assemble()

            a = BilinearForm(H_H, symmetric=True, symmetric_storage=True) # define the  inner product on the high order space
            a += InnerProduct(u_H,v_H) * dx
            a.Assemble()

            #mass matrix with mixed basis functions
            a_mixed = BilinearForm(trialspace=H_h, testspace=H_H)
            a_mixed += u_h*v_H*dx
            a_mixed.Assemble()

            #compute inverses 
            a_inv = a.mat.Inverse(freedofs=H_H.FreeDofs())
            m_inv = m.mat.Inverse(freedofs=H_h.FreeDofs())

            #create Matrices for GEV problem

            #c is the low order galerkin matrix
            c = BilinearForm(H_h, symmetric=True, symmetric_storage=False)
            c += InnerProduct(u_H,v_H) * dx
            c.Assemble()

            # b is the Matrix that lets us compute the norm of the Riesz representative
            # m==m.t, m_inv == m_inv.T
            b = c.mat.T @m_inv @ a_mixed.mat.T @ a_inv @ a_mixed.mat @ m_inv @ c.mat

            tmp1 = w_h.vec.CreateVector()
            tmp2 = w_h.vec.CreateVector()
            tmp3 = w_h.vec.CreateVector()
            tmp4 = w_h.vec.CreateVector()

            B = sp.sparse.linalg.LinearOperator((b.height,b.width), matvec_b)
            C = sp.sparse.linalg.LinearOperator((c.mat.height,c.mat.width), matvec_c)     
            #The matrices Involved are Symmetric, so the symmetric solver is used
            #look for largest Eigenvalue of Bx = λCx, since ARPACK is more efficient for large EV's
            lam = sp.sparse.linalg.eigsh(B, k=1, M=C, which='LM', return_eigenvectors=False)
            print(lam)
            #if FEM space is complex need to take absolut value (the EV's have no imaginary part, but are still datatype complex)        
            #lam = np.abs(lam)
            #1/λ is the smallest EV of Cx = λBx
            minEV[i-lowest_high_Order,k] = 1/lam[0]

            print("\n")

    print(minEV)

    return minEV
