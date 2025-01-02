from ngsolve import *
from netgen.occ import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

#l is number of meshwidths, the n-th meshwidth is 1/(2^(n-1))
l = 7
meshwidths = np.ones(l)
for h in range(l-1):
    meshwidths[h+1] = meshwidths[h]/2

print(meshwidths)
#order for which the elements corresponds to the whitney forms H1,HCurl = 1, HDiv ,L2=0
lowest_low_order = 1
# amount of orders for the low-Order space (set to 1 if not interested in higher low orders)
low_orders = 1
# largest low order to test
highest_low_order = lowest_low_order + low_orders

#amount of orders for the high-Order space
high_orders = 3 
#array storing the minimal Eigenvalue for given orders and meshwidth
minEV = np.zeros((low_orders,high_orders,l))

for j in range(lowest_low_order , highest_low_order):
    #high order needs to be greater than low order
    lowest_high_Order = j + 1
    #largest high order to test
    highest_high_order = lowest_high_Order + high_orders
    for i in range(lowest_high_Order, highest_high_order):
        #reset mesh for new high Order
        netgen_mesh =unit_cube.GenerateMesh(maxh=1)
        mesh=Mesh(netgen_mesh)

        print("polynomal order of low order space:", j, ", polynomal order of high order space:", i)
        for k in range(l):
            print("h=",meshwidths[k], "\n")

            #set function space, for l=0 its normal lagrangian finite elements
            #need to compress to remove DOFs of unrefined mesh after refinement
            H_h = Compress(H1(mesh, order = j, complex=False)) # main function space
            H_H = Compress(H1(mesh, order = i, complex=False)) # high order Function space for Riesz representative  

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

            a = BilinearForm(H_H, symmetric=True, symmetric_storage=True) # define the H1 inner product on the high order space
            a += grad(u_H) * grad(v_H) * dx
            a += u_H*v_H * dx 
            a.Assemble()

            #mass matrix with mixed basis functions
            a_mixed = BilinearForm(trialspace=H_h, testspace=H_H)
            a_mixed += u_h*v_H*dx
            a_mixed.Assemble()

            #compute inverses 
            a_inv = a.mat.Inverse()
            m_inv = m.mat.Inverse()

            #create Matrices for GEV problem

            #c is the low order galerkin matrix
            c = BilinearForm(H_h, symmetric=True, symmetric_storage=False)
            c += grad(u_h) * grad(v_h) * dx
            c += u_h*v_h * dx 
            c.Assemble()

            # b is the Matrix that lets us compute the norm of the Riesz representative
            # m==m.t, m_inv == m_inv.T
            b = c.mat.T @m_inv @ a_mixed.mat.T @ a_inv @ a_mixed.mat @ m_inv @ c.mat
            B = b.ToDense().NumPy()

            #C = c.mat.ToDense().NumPy()
            #use scipy sparse matrix format to avoid massive memory usage
            rows,cols,vals = c.mat.COO()
            C = sp.sparse.csr_matrix((vals,(rows,cols)))

            #The matrices Involved are Symmetric, so the symmetric solver is used
            #look for largest Eigenvalue of Bx = λCx, since ARPACK is more efficient for large EV's
            lam = sp.sparse.linalg.eigsh(B, k=1, M=C, which='LM', return_eigenvectors=False)
            print(lam)
            #if FEM space is complex need to take absolut value (the EV's have no imaginary part, but are still datatype complex)        
            #lam = np.abs(lam)
            #1/λ is the smallest EV of Cx = λBX
            minEV[j-lowest_low_order,i-lowest_high_Order,k] = 1/lam[0]

            #uniformly refines mesh, halving meshwidth
            mesh.Refine()
            print("\n")

print(minEV)

#np.save('d{d}l{l}_minEV'.format(d=3,l=0),minEV)
np.save('/cluster/home/rschoenholze/Bsc_Thesis/data/d{d}l{l}_minEV'.format(d=3,l=0),minEV)

symbols = ['o-','h-.','*:','+-']

ref_val = 1/3

#minimal Ev
for j in range(lowest_low_order, highest_low_order):
    fig, ax = plt.subplots()
    plt.grid(visible=True)
    plt.title(label="d=3, l=0, low order=%i" %j)
    plt.loglog(meshwidths,np.ones(l) * ref_val,'--k', label="$1/3$")
    plt.xlabel('meshwidth h')
    plt.ylabel('minimal Eigenvalue')

    lowest_high_Order = j + 1
    highest_high_order = lowest_high_Order + high_orders
    for i in range(lowest_high_Order,highest_high_order):
        plt.loglog(meshwidths,minEV[j-lowest_low_order,i-lowest_high_Order,:], symbols[i-lowest_high_Order], label="high order=%i"%i)

    plt.legend()
    #plt.savefig("../plots/higherOrders/d3l0/d3l0_minEV_o%i.pdf" %j)
    plt.savefig("/cluster/home/rschoenholze/Bsc_Thesis/higherOrders/d3l0/d3l0_minEV_o%i.pdf" %j)

# c_S = sqrt(minEV)
for j in range(lowest_low_order, highest_low_order):
    fig, ax = plt.subplots()
    plt.grid(visible=True)
    plt.title(label="d=3, l=0, low order=%i" %j)
    plt.loglog(meshwidths,np.ones(l) * sqrt(ref_val),'--k', label="$\sqrt{1/3}$")
    plt.xlabel('meshwidth h')
    plt.ylabel(r'$c_{S}$')

    lowest_high_Order = j + 1
    highest_high_order = lowest_high_Order + high_orders
    for i in range(lowest_high_Order,highest_high_order):
        plt.loglog(meshwidths,np.sqrt(minEV[j-lowest_low_order,i-lowest_high_Order,:]), symbols[i-lowest_high_Order], label="high order=%i"%i)

    plt.legend()
    #plt.savefig("../plots/higherOrders/d3l0/d3l0_constant_o%i.pdf" %j)
    plt.savefig("/cluster/home/rschoenholze/Bsc_Thesis/higherOrders/d3l0/d3l0_constant_o%i.pdf" %j)

#convergence rate for minimal EV

for j in range(lowest_low_order, highest_low_order):
    fig, ax = plt.subplots()
    plt.grid(visible=True)
    plt.xlabel('meshwidth h')
    plt.ylabel(r"$ (\lambda - \lambda_{ref}) * \lambda_{ref}^{-1} $")
    plt.title(r"convergence to $\lambda_{ref}=\frac{1}{3}$, low order=%i" %j) 
    plt.loglog(meshwidths,np.power(meshwidths,2),'--k', label=r'$\mathcal{O}(h^{2})$')

    lowest_high_Order = j + 1
    highest_high_order = lowest_high_Order + high_orders
    for i in range(lowest_high_Order,highest_high_order):
        plt.loglog(meshwidths,(minEV[j-lowest_low_order,i-lowest_high_Order,:] - np.ones(l)*ref_val)/ref_val, symbols[i-lowest_high_Order], label="high order=%i"%i)

    plt.legend()
    #plt.savefig("../plots/higherOrders/d3l0/d3l0_convergence_theory_o%i.pdf" %j)
    plt.savefig("/cluster/home/rschoenholze/Bsc_Thesis/higherOrders/d3l0/d3l0_convergence_theory_o%i.pdf" %j)

#convergence rate for minimal EV
for j in range(lowest_low_order, highest_low_order):
    fig, ax = plt.subplots()
    plt.grid(visible=True)
    plt.xlabel('meshwidth h')
    plt.ylabel(r"$ (\lambda - \lambda_{ref}) * \lambda_{ref}^{-1} $")
    plt.title(r"convergence to $\lambda_{ref}=\lambda_{min,finest}$, low order=%i" %j) 
    plt.loglog(meshwidths,np.power(meshwidths,2),'--k', label=r'$\mathcal{O}(h^{2})$')

    lowest_high_Order = j + 1
    highest_high_order = lowest_high_Order + high_orders
    ref_val = minEV[j-lowest_low_order,:,l-1]
    for i in range(lowest_high_Order,highest_high_order):
        plt.loglog(meshwidths,(minEV[j-lowest_low_order,i-lowest_high_Order,:] - np.ones(l)*ref_val[i-lowest_high_Order])/ref_val[i-lowest_high_Order], symbols[i-lowest_high_Order], label="high order=%i"%i)

    plt.legend()
    #plt.savefig("../plots/higherOrders/d3l0/d3l0_convergence_FineEV_o%i.pdf" %j)
    plt.savefig("/cluster/home/rschoenholze/Bsc_Thesis/higherOrders/d3l0/d3l0_convergence_FineEV_o%i.pdf" %j)