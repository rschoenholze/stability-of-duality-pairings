from ngsolve import *
from netgen.occ import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import dualMeshInnerProducts as dmIP

def getLowestPolynomialOrder(dim, diffform, swapHDHC):
    '''returns the polynomial order of the Whitneyforms in NGSolve.
    dim needs to be 2 or 3, difform determines the integer l for the l-form, difform <= dim,
    swapHDHC is a bool, when false: in 2D 1-forms are H(div) elements,
    when True 1-forms in 2D are H(curl) elements.
    
    In NGSolve the lowest polynomial orders are:
    H1: 1
    HDiv: 0
    HCurl: 1
    L2: 0
    '''
    match dim:
        case 2:
            if swapHDHC == False:
                orders = [1,0,0]
            else:
                orders = [1,1,0]
        case 3:
            orders = [1,1,0,0]


    return orders[diffform]

def getFunctionSpace(dim, diffform, swapHDHC):
    '''returns the function space corresponding to diffform.
    dim needs to be 2 or 3, difform determines the integer l for the l-form, difform <= dim,    
    swapHDHC is a bool, when false: in 2D 1-forms are H(div) elements,
    when True 1-forms in 2D are H(curl) elements.'''
    match dim:
        case 2:
            if swapHDHC == False:
                differentialforms = [H1, HDiv, L2]
            else:
                differentialforms = [H1, HCurl, L2]
        case 3:
            differentialforms = [H1, HCurl, HDiv, L2]

    
    return differentialforms[diffform]

def getInnerProduct(dim, difform, swapHDHC):
    '''returns the Innerproduct corresponding to diffform.
    dim needs to be 2 or 3, difform determines the integer l for the l-form, difform <= dim,    
    swapHDHC is a bool, when false: in 2D 1-forms are H(div) elements,
    when True 1-forms in 2D are H(curl) elements.'''
    match difform:
        case 0:
            return lambda a,b: a * b + grad(a) * grad(b)
        case 1:
            if (dim == 2 and swapHDHC == False):
                return lambda a,b: a * b + div(a) * div(b)
            elif (dim == 2 and swapHDHC == True) or (dim == 3):
                return lambda a,b: a * b + curl(a) * curl(b)
        case 2:
            if dim == 3:
                return lambda a,b: a * b + div(a) * div(b)
            else:
                return lambda a,b: a * b 
        case 3:
            return lambda a,b: a * b 

def createMesh(mw, numRef, mt, b, d):
    '''returns a mesh.
    mw is the desired meshwidth, numRef is the number of times 1 would need to be halved to arrive at the meshwidth,
    mt is the meshtype (either str for structured or unstr for unstructured, anything else will return an unstructured mesh),
    b is the dirichlet boundary conditions, d is the dimension (needs to be 2 or 3)'''
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
            #these arguments create a mesh, with an average meshwidth close to mw
            netgen_mesh = domain.GenerateMesh(maxh=mw, segmentsperedge=(numRef + offset)+1.2, grading=0.1)
            mesh=Mesh(netgen_mesh)
    
    return mesh

def estimate_stability(num_mw = 4, bnd = '', diff_form = 0,dim = 2, mesh_type = "str", num_HO = 3, Swap_HCurl_HDiv = False):
    '''returns a num_mx by num_HO matrix containing the minimal Eigenvalues of the general eigenvalue problem which estimates 
    the Inf-Sup condition, to get the Inf-Sup constants, take the square root of the return value.

    num_mw is the number of meshwidths, starting at 1 and halving the prevoius meshwidth to get the next one, 
    if boundary conditions are specified starting meshwidth is 0.5.

    bnd is a string containing the boundary conditions, in the same style as NGSolve.

    diff_form determines for which whitney l-form stability is estimated, needs to be smaller or equal to dim.

    dim is the dimension, either 2 or 3.

    mesh_type determines wether the mesh is structured (argument needs to be "str"), or unstructured ("unstr" or anything else).

    num_HO is the number of higher polynomial orders for which the Riesz representative should be approximated.

    Swap_HCurl_HDiv, if False 1-forms in 2D will be H(div) elements 
    when True 1-forms in 2D are H(curl) elements.'''
    
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

    #polynomial order for which the elements corresponds to the whitney forms in NGSolve
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
        for k in range(l):
            mesh = createMesh(mw = meshwidths[k], numRef=k, mt=mesh_type, b=bnd, d=dim)

            #need to compress to remove DOFs of unrefined mesh after refinement
            # main function space
            H_h = Compress(fes(mesh, order = low_order, dirichlet=bnd, RT=True, type1=True)) 
            #reference vector for matvec wrapper
            w_h = GridFunction(H_h) 
            # high order Function space for Riesz representative 
            H_H = Compress(fes(mesh, order = i, dirichlet=bnd, RT=True, type1=True))  

            #low order
            u_h = H_h.TrialFunction() # define the trial function
            v_h = H_h.TestFunction() # define the test function
            #High order
            u_H = H_H.TrialFunction() # define the trial function
            v_H= H_H.TestFunction() # define the test function

            #define m with the L2 inner product on the low order space
            m = BilinearForm(H_h, symmetric=True, symmetric_storage=True) 
            m += u_h*v_h * dx 
            m.Assemble()

            # define the  inner product on the high order space
            a = BilinearForm(H_H, symmetric=True, symmetric_storage=True) 
            a += InnerProduct(u_H,v_H) * dx
            a.Assemble()

            #mass matrix with mixed basis functions, of different polynomial orders
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

            #vectors needed for the matrix vector products 
            tmp1 = w_h.vec.CreateVector()
            tmp2 = w_h.vec.CreateVector()
            tmp3 = w_h.vec.CreateVector()
            tmp4 = w_h.vec.CreateVector()

            #to use scipy solvers NGSolve matrices need to be wrapped in linear Operators
            B = sp.sparse.linalg.LinearOperator((b.height,b.width), matvec_b)
            C = sp.sparse.linalg.LinearOperator((c.mat.height,c.mat.width), matvec_c)     
            #The matrices Involved are Symmetric, so the symmetric solver is used
            #look for largest Eigenvalue of Bx = λCx, since ARPACK is more efficient for large EV's
            lam = sp.sparse.linalg.eigsh(B, k=1, M=C, which='LM', return_eigenvectors=False)
            #1/λ is the smallest EV of Cx = λBx
            minEV[i-lowest_high_Order,k] = 1/lam[0]

    return minEV


#TODO str mesh needs to be deleted for dual inner prod
def estimate_stability_dual_Mesh(num_mw = 4, bnd = '', diff_form = 0, mesh_type = "str", num_HO = 3, Swap_HCurl_HDiv = False):
    '''returns a num_mx by num_HO matrix containing the minimal Eigenvalues of the general eigenvalue problem which estimates 
    the Inf-Sup condition, to get the Inf-Sup constants, take the square root of the return value. 

    The dual space is approximated by the dual mesh approach. Dimension is always 2.

    num_mw is the number of meshwidths, starting at 1 and halving the prevoius meshwidth to get the next one, 
    if boundary conditions are specified starting meshwidth is 0.5.

    bnd is a string containing the boundary conditions, in the same style as NGSolve.

    diff_form determines for which whitney l-form stability is estimated, needs to be smaller or equal to 2.

    mesh_type determines wether the mesh is structured (argument needs to be "str"), or unstructured ("unstr" or anything else),
    if the mesh is structured, it will create a file in the current woring directory.

    num_HO is the number of higher polynomial orders for which the Riesz representative should be approximated.

    Swap_HCurl_HDiv, if False 1-forms in 2D will be H(div) elements, if True they will be H(curl) elements.'''
    
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

    #polynomial order for which the elements corresponds to the whitney forms in NGSolve
    low_order = getLowestPolynomialOrder(2, diff_form, Swap_HCurl_HDiv)

    #amount of orders for the high-Order space
    high_orders = num_HO 
    #array storing the minimal Eigenvalue for given orders and meshwidth
    minEV = np.zeros((high_orders,l))

    fes = getFunctionSpace(2,diff_form,Swap_HCurl_HDiv)

    InnerProduct = getInnerProduct(2,diff_form,Swap_HCurl_HDiv)

    L2innerProd_dual_mesh = [dmIP.H1_Primal_Dual_L2IP, dmIP.HDiv_Primal_Dual_L2IP, dmIP.L2_Primal_Dual_L2IP]
    dual_inner_Prod = L2innerProd_dual_mesh[diff_form]

    #high order needs to be greater than low order
    lowest_high_Order = low_order + 1
    #largest high order to test
    highest_high_order = lowest_high_Order + high_orders
    for i in range(lowest_high_Order, highest_high_order):
        for k in range(l):
            mesh = createMesh(mw = meshwidths[k], numRef=k, mt=mesh_type, b=bnd, d=2)

            #Barycentric Refinement crashes Programm if coarse mesh information is not deleted,
            #only relevant for structured mesh, save, delete and load the mesh, 
            #this will delete coarse mesh informtaion
            if mesh_type=='str':
                mesh.ngmesh.Save("mesh")
                del mesh
                mesh = Mesh("mesh.vol.gz")
                mesh._updateBuffers()

            #dual_inner_Prod changes the mesh, need to pass a copy
            netgen_dual = mesh.ngmesh.Copy()
            dual_mesh=Mesh(netgen_dual)

            #need to compress to remove DOFs of unrefined mesh after refinement
            # main function space
            H_h = Compress(fes(mesh, order = low_order, dirichlet=bnd, RT=True, type1=True)) 
            #reference vector for matvec wrapper
            w_h = GridFunction(H_h) 
            # high order Function space for Riesz representative
            H_H = Compress(fes(mesh, order = i, dirichlet=bnd, RT=True, type1=True))   

            #low order
            u_h = H_h.TrialFunction() # define the trial function
            v_h = H_h.TestFunction() # define the test function
            #High order
            u_H = H_H.TrialFunction() # define the trial function
            v_H= H_H.TestFunction() # define the test function

            #the L2 inner of the primal space and the corresponding dual space constructed on the dual mesh
            m = dual_inner_Prod(dual_mesh, Swap_HCurl_HDiv)

            # define the  inner product on the high order space
            a = BilinearForm(H_H, symmetric=True, symmetric_storage=True) 
            a += InnerProduct(u_H,v_H) * dx
            a.Assemble()

            #mass matrix with mixed basis functions, of different polynomial orders
            a_mixed = BilinearForm(trialspace=H_h, testspace=H_H)
            a_mixed += u_h*v_H*dx
            a_mixed.Assemble()

            #compute inverses 
            a_inv = a.mat.Inverse(freedofs=H_H.FreeDofs())
            m_inv = m.Inverse(freedofs=H_h.FreeDofs())

            #create Matrices for GEV problem

            #c is the low order galerkin matrix
            c = BilinearForm(H_h, symmetric=True, symmetric_storage=False)
            c += InnerProduct(u_H,v_H) * dx
            c.Assemble()

            # b is the Matrix that lets us compute the norm of the Riesz representative
            b = c.mat.T @m_inv.T @ a_mixed.mat.T @ a_inv @ a_mixed.mat @ m_inv @ c.mat

            #vectors needed for the matrix vector products 
            tmp1 = w_h.vec.CreateVector()
            tmp2 = w_h.vec.CreateVector()
            tmp3 = w_h.vec.CreateVector()
            tmp4 = w_h.vec.CreateVector()

            #to use scipy solvers NGSolve matrices need to be wrapped in linear Operators
            B = sp.sparse.linalg.LinearOperator((b.height,b.width), matvec_b)
            C = sp.sparse.linalg.LinearOperator((c.mat.height,c.mat.width), matvec_c)     
            #The matrices Involved are Symmetric, so the symmetric solver is used
            #look for largest Eigenvalue of Bx = λCx, since ARPACK is more efficient for large EV's
            lam = sp.sparse.linalg.eigsh(B, k=1, M=C, which='LM', return_eigenvectors=False)
            #1/λ is the smallest EV of Cx = λBx
            minEV[i-lowest_high_Order,k] = 1/lam[0]

    if mesh_type=='str':
        ...

    return minEV
