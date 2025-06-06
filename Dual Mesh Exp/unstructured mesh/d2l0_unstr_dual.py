from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def matvec_b(v):
    tmp1.data = v
    tmp2.data = b * tmp1
    return tmp2.FV().NumPy()


def matvec_c(v):
    tmp3.data = v
    tmp4.data = c.mat * tmp3
    return tmp4.FV().NumPy()

def H1_Primal_Dual_L2IP(mesh):

    #Setup
    pv = mesh.vertices
    nv = len(pv)

    #Barycentric Refinement
    mesh.ngmesh.SplitPowellSabin()

    #for fine to coarse mapping
    fes_primal = Compress(H1(mesh, order=1, dirichlet="bottom|right|top|left"))

    H1_u = fes_primal.TrialFunction()
    H1_v = fes_primal.TestFunction()

    #for mapping to dual
    fes_dual = Compress(L2(mesh, order=0, dirichlet="bottom|right|top|left"))

    L2_u = fes_dual.TrialFunction()
    L2_v = fes_dual.TestFunction()

    #L2 inner product of H1 and L2 basis function on barycentrically refined mesh
    H1_L2_L2IP = BilinearForm(trialspace=fes_primal, testspace=fes_dual) #fes_dual)
    H1_L2_L2IP += H1_u * L2_v * dx

    H1_L2_L2IP.Assemble()

    rows,cols,vals  = H1_L2_L2IP.mat.COO()

    mat = sp.sparse.csr_matrix((vals,(rows,cols)))

    # mapping to dual 
    data_length = 0

    for v in pv:
        data_length += len(v.faces)

    data1 = np.zeros(data_length)
    row_ind1 = np.zeros(data_length, dtype=int)
    col_ind1 = np.zeros(data_length, dtype=int)

    j = 0

    for v in pv:

        coeff =  1

        for el in v.elements:
            #print(f)
            el_dofNr = fes_dual.GetDofNrs(el)
            #print(dofNr)
            row_ind1[j] = el_dofNr[0]
            col_ind1[j] = v.nr
            data1[j] = coeff
            j += 1
    ne = mesh.ne
    dual_map = sp.sparse.csr_matrix((data1,(row_ind1,col_ind1)))
    
    # mapping to coarse primal

    data_length = 0

    for par in pv:
        for el in par.elements:
            data_length += 3

    data = np.zeros(data_length)
    row_ind = np.zeros(data_length, dtype=int)
    col_ind = np.zeros(data_length, dtype=int)

    j = 0

    for par in pv:
        par_dofnr = fes_primal.GetDofNrs(par)[0]
        par_nf = len(par.faces)
        for el in par.elements:
            ele = mesh.__getitem__(el)
            for v in ele.vertices:
                ver = mesh.__getitem__(v)
                child_dofnr = fes_primal.GetDofNrs(v)[0]
                row_ind[j] = par_dofnr
                col_ind[j] = child_dofnr
                #coarse vertices
                if par_dofnr == child_dofnr:
                    data[j] = 1/par_nf
                #barycenters
                elif len(ver.faces) == 6:
                    data[j] = 1/6
                #boundry elements
                elif len(ver.faces) == 2:
                    data[j] = 1/2
                else:
                    data[j] = 1/4
                j += 1

    primal_map = sp.sparse.csr_matrix((data,(row_ind,col_ind)))

    dual_L2IP =  dual_map.T @ mat @ primal_map.T 

    coo = dual_L2IP.tocoo(copy=True)

    NG_dual = la.SparseMatrixdouble.CreateFromCOO(coo.row,coo.col,coo.data, nv, nv)

    return NG_dual

#l is number of meshwidths, the n-th meshwidth is 1/(2^(n-1))
l = 6
meshwidths = np.ones(l) * 0.5
for h in range(l-1):
    meshwidths[h+1] = meshwidths[h]/2

print(meshwidths)
#order for which the elements corresponds to the whitney forms
low_order = 1

#amount of orders for the high-Order space
high_orders = 3 
#array storing the minimal Eigenvalue for given orders and meshwidth
minEV = np.zeros((high_orders,l))

lowest_high_Order = low_order + 1
    #largest high order to test
highest_high_order = lowest_high_Order + high_orders

for i in range(lowest_high_Order, highest_high_order):
    print("polynomal order of low order space:", low_order, ", polynomal order of high order space:", i)
    for k in range(l):
        mw = meshwidths[k]
        print("h=",mw)
        #Draw(mesh)

        #increase segmentsper edge by 1 since we start at 0.5 mw
        netgen_mesh = unit_square.GenerateMesh(maxh=mw, segmentsperedge=(1+k)+1.2, grading=0.1)
        mesh=Mesh(netgen_mesh)

        netgen_dual = netgen_mesh.Copy()
        dual_mesh=Mesh(netgen_dual)

        #actual meshwidth
        elvol = Integrate(CoefficientFunction(1),mesh,element_wise=True)
        mesh_h = [(2*vol)**(1/2) for vol in elvol]
        print("actual meshwidth range",min(mesh_h),max(mesh_h), "\n")

        #set function space, for l=0 its normal lagrangian finite elements
        #need to compress to remove DOFs of unrefined mesh after refinement
        H_h = Compress(H1(mesh, order = low_order, complex=False, dirichlet="bottom|right|top|left")) # main function space
        w_h = GridFunction(H_h) #define vector for matvec wrapper
        H_H = Compress(H1(mesh, order = i, complex=False, dirichlet="bottom|right|top|left")) # high order Function space for Riesz representative  

        print("# DoFs of low order space:", H_h.ndof, ", # DoFs of high order space:", H_H.ndof)

        #low order
        u_h = H_h.TrialFunction() # define the trial function
        v_h = H_h.TestFunction() # define the test function
        #High order
        u_H = H_H.TrialFunction() # define the trial function
        v_H= H_H.TestFunction() # define the test function

        a = BilinearForm(H_H, symmetric=True, symmetric_storage=False) # define the H1 inner product on the high order space
        a += grad(u_H) * grad(v_H) * dx
        a += u_H*v_H * dx 
        a.Assemble()

        #mass matrix with mixed basis functions
        a_mixed = BilinearForm(trialspace=H_h, testspace=H_H)
        a_mixed += u_h*v_H*dx
        a_mixed.Assemble()

        m_dual = H1_Primal_Dual_L2IP(dual_mesh)

        #compute inverses 
        #to consider BC set freedofs to H_H.FreeDofs() 
        a_inv = a.mat.Inverse() #freedofs=H_H.FreeDofs()

        #create Matrices for GEV problem

        #c is the low order galerkin matrix
        c = BilinearForm(H_h, symmetric=True, symmetric_storage=False)
        c += grad(u_h) * grad(v_h) * dx
        c += u_h*v_h * dx 

        c.Assemble()
        #to consider BC set freedofs to H_h.FreeDofs() 
        m_dual_inv = m_dual.Inverse() #freedofs=H_h.FreeDofs())

        # b is the Matrix that lets us compute the norm of the Riesz representative
        b = c.mat.T @ m_dual_inv.T @ a_mixed.mat.T @ a_inv @ a_mixed.mat @ m_dual_inv @ c.mat

        tmp1 = w_h.vec.CreateVector()
        tmp2 = w_h.vec.CreateVector()

        tmp3 = w_h.vec.CreateVector()
        tmp4 = w_h.vec.CreateVector()

        B = sp.sparse.linalg.LinearOperator((b.height,b.width), matvec_b)
        C = sp.sparse.linalg.LinearOperator((c.mat.height,c.mat.width), matvec_c)            

        #look for largest Eigenvalue of Bx = λCx, since ARPACK is more efficient for large EV's
        lam = sp.sparse.linalg.eigsh(B, k=1, M=C, which='LM', return_eigenvectors=False)
        print(lam)
        #1/λ is the smallest EV of Cx = λBx
        minEV[i-lowest_high_Order,k] = 1/lam[0]

        print("\n")

print(minEV)

np.save('d2l0_unstr_dual',minEV)




