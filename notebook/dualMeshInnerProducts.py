from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

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
        #skip v if boundry vertex

        bnd_vertex = False
        for bnd_el in mesh.Elements(BND):
            if v in bnd_el.vertices:
                bnd_vertex = True
                break
        
        if bnd_vertex == True:
            coeff = 0
        else:
            coeff =  1#/len(v.elements)

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
        #print(v)
        par_dofnr = fes_primal.GetDofNrs(par)[0]
        par_nf = len(par.faces)
        for el in par.elements:
            ele = mesh.__getitem__(el)
            for v in ele.vertices:
                ver = mesh.__getitem__(v)
                child_dofnr = fes_primal.GetDofNrs(v)[0]
                #print(dofnr)
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


def L2_Primal_Dual_L2IP(mesh):

    pv = mesh.vertices
    nv = len(pv)
    #Setup

    #Barycentric Refinement
    mesh.ngmesh.SplitPowellSabin()

    #for fine to coarse mapping
    fes_primal = Compress(L2(mesh, order=0, dirichlet="bottom|right|top|left"))

    L2_u = fes_primal.TrialFunction()

    #for mapping to dual
    fes_dual = Compress(H1(mesh, order=1, dirichlet="bottom|right|top|left"))

    H1_v = fes_dual.TestFunction()

    #L2 inner product of H1 and L2 basis function on barycentrically refined mesh
    L2_H1_L2IP = BilinearForm(trialspace=fes_primal, testspace=fes_dual) #fes_dual)
    L2_H1_L2IP += L2_u * H1_v * dx

    L2_H1_L2IP.Assemble()
    #print(L2_H1_L2IP.mat.shape)

    rows,cols,vals  = L2_H1_L2IP.mat.COO()

    mat = sp.sparse.csr_matrix((vals,(rows,cols)))

    # mapping to dual 

    data_length = mesh.ne * 7
    data = np.zeros(data_length)
    row_ind = np.zeros(data_length, dtype=int)
    col_ind = np.zeros(data_length, dtype=int)

    j = 0

    for el in mesh.Elements():
        parent = mesh.ngmesh.MacroElementNr(el.nr)
        for v in el.vertices:
            ver = mesh.__getitem__(v)
            dofnr = fes_dual.GetDofNrs(ver)
            row_ind[j] = parent
            col_ind[j] = dofnr[0]
            data[j] = 1/len(ver.elements)
            j += 1

    dual_map = sp.sparse.csr_matrix((data,(row_ind,col_ind)))
    # mapping to coarse primal
    #L2 fine to coarse mapping

    data_length = mesh.ne
    data = np.zeros(data_length)
    row_ind = np.zeros(data_length, dtype=int)
    col_ind = np.zeros(data_length, dtype=int)

    j = 0

    for el in mesh.Elements():
        parent = mesh.ngmesh.MacroElementNr(el.nr)
        for dofnr in fes_primal.GetDofNrs(el):
            row_ind[j] = parent
            col_ind[j] = dofnr
            data[j] = 1
            j += 1

    primal_map = sp.sparse.csr_matrix((data,(row_ind,col_ind)))
    
    #TODO make sure this constellation of shapes, and what they represent make sense
    dual_L2IP = dual_map @ mat @ primal_map.T

    coo = dual_L2IP.tocoo(copy=True)

    NG_dual = la.SparseMatrixdouble.CreateFromCOO(coo.row,coo.col,coo.data, mesh.ne//6, mesh.ne//6)

    return NG_dual

#############################################################################
#Helper functions for inner product on the dual mesh for differential 1 forms
def edge_length(e, mesh):
    p1 = mesh.__getitem__(e.vertices[1]).point
    p0 = mesh.__getitem__(e.vertices[0]).point
    e_len = sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
    return e_len

def append_next_edge(edges, prev_edge, mesh, vi, v2):
    for el in prev_edge.elements:
        el = mesh.__getitem__(el)
        for next_e in el.edges:
            next_e = mesh.__getitem__(next_e)
            if (next_e not in edges) and (vi in next_e.vertices) and (v2 not in next_e.vertices):
                edges.append(next_e)
                return
            
def get_ref_edge(central_edge, mesh, vi, v2):
    for el in central_edge.elements:
        el = mesh.__getitem__(el)
        for next_e in el.edges:
            next_e = mesh.__getitem__(next_e)
            if (vi in next_e.vertices) and (v2 in next_e.vertices):
                return next_e
###############################################################################

def HDiv_Primal_Dual_L2IP(mesh):
    pe = mesh.edges
    ne = mesh.nedge
    pv = mesh.vertices
    nv = len(pv)
    #Setup

    #for fine to coarse mapping

    elnrs = np.ones((ne,2),dtype=int)
    for e in mesh.edges:
        el1 = mesh.__getitem__(e.elements[0])
        elnrs[e.nr,:] = el1.nr
        if len(e.elements) == 2:
            el2 = mesh.__getitem__(e.elements[1])
            elnrs[e.nr,1] = el2.nr

    #print(elnrs[:,:])

    #Barycentric Refinement
    mesh.ngmesh.SplitPowellSabin()

    #for fine to coarse mapping
    fes_primal = Compress(HCurl(mesh, order=1, type1=True, dirichlet="bottom|right|top|left"))

    Hdiv_u = fes_primal.TrialFunction()
    Hdiv_v = fes_primal.TestFunction()

    #for mapping to dual
    fes_dual =  Compress(HDiv(mesh, order=0, RT=True, dirichlet="bottom|right|top|left"))

    Hdual_u = fes_dual.TrialFunction()
    Hdual_v = fes_dual.TestFunction()

    #L2 inner product of H1 and L2 basis function on barycentrically refined mesh
    HD_HC_L2IP = BilinearForm(trialspace=fes_primal, testspace=fes_dual) #fes_dual)
    HD_HC_L2IP += Hdiv_u * Hdual_v * dx

    HD_HC_L2IP.Assemble()
    #print(HD_HC_L2IP.mat.shape)

    rows,cols,vals  = HD_HC_L2IP.mat.COO()

    mat = sp.sparse.csr_matrix((vals,(rows,cols)))
    #print(mat.shape)

    # mapping to dual ---------------------------------------------------------------------------------------------------------
    data_length = 0
    for e in pe:
        for v in e.vertices:
            v = mesh.__getitem__(v)
            data_length += len(v.edges) + 4

    data = np.zeros(data_length)
    row_ind = np.zeros(data_length, dtype=int)
    col_ind = np.zeros(data_length, dtype=int)

    j = 0

    for e in pe:
        #getting accesible information
        parent = e.nr  
        v0 = e.vertices[0]  
        v1 = e.vertices[1] 
        v0 = mesh.__getitem__(v0)
        v1 = mesh.__getitem__(v1)

        bnd_edge = False
        if elnrs[parent, 0] == elnrs[parent,1]:
            bnd_edge = True

        #finding the central dual edges of the reference edge
        central_edges = []

        for el0 in v0.elements:
            for el1 in v1.elements:
                el0 = mesh.__getitem__(el0)
                el1 = mesh.__getitem__(el1)
                intersect = tuple(set(el0.edges).intersection(el1.edges))
                if intersect != ():
                    e = mesh.__getitem__(intersect[0])
                    central_edges.append(e)

        #finding vertex in edge midpoint
        if len(central_edges) == 2:
            v2 = tuple(set(central_edges[0].vertices).intersection(central_edges[1].vertices))[0]
            v2 = mesh.__getitem__(v2)
        else:
            for v in central_edges[0].vertices:
                v = mesh.__getitem__(v)
                if len(v.faces) == 2:
                    v2 = v

        #setting the coefficients for the edges
        verts = [v0,v1]
        sign = [1, -1]
        if bnd_edge == False:
            for k in range(2):
                vi = verts[k]
                Nc = len(vi.faces)/2
                Ne = len(vi.edges)

                bnd_vertex = False
                for bnd_el in mesh.Elements(BND):
                    if vi in bnd_el.vertices:
                        bnd_vertex = True

                s = sign[k]
                edges = []

                if bnd_vertex == False:
                    #INTERNAL vertex ---------------------------------------------------------------------
                    edges = [central_edges[k]]
                    
                    i = 0    
                    for e_i in edges:            
                        
                        #calculate edge length
                        e_len = edge_length(e_i, mesh)
                        coeff = (Nc - i)/(2*e_len*Nc) 
                        row_ind[j] = parent
                        col_ind[j] = fes_dual.GetDofNrs(e_i)[0]
                        data[j] = coeff *s
                        j += 1

                        append_next_edge(edges, e_i, mesh, vi, v2)
                        i += 1
                else:
                    #BND vertex --------------------------------------------------------------------------
                    #deal with reference edge (part of original edge)
                    ref_edge = get_ref_edge(central_edges[0], mesh, vi, v2)
                    e_len = edge_length(ref_edge, mesh)
                    coeff = (2 - Nc)/(2*e_len*Nc) 
                    row_ind[j] = parent
                    col_ind[j] = fes_dual.GetDofNrs(ref_edge)[0]
                    data[j] = coeff *s
                    j += 1

                    #deal with central edge
                    e_len = edge_length(central_edges[k], mesh)
                    coeff = (1)/(2*e_len) 
                    row_ind[j] = parent
                    col_ind[j] = fes_dual.GetDofNrs(central_edges[k])[0]
                    data[j] = coeff *s
                    j += 1

                    #edges with counterclockwise lower index than ref edge
                    append_next_edge(edges, central_edges[k-1], mesh, vi, v2)
                    for e_i in edges:            
                        #calculate edge length
                        e_len = edge_length(e_i, mesh)
                        coeff = (1-Nc)/(e_len*Nc) 
                        row_ind[j] = parent
                        col_ind[j] = fes_dual.GetDofNrs(e_i)[0]
                        data[j] = coeff * s 
                        j += 1

                        append_next_edge(edges, e_i, mesh, vi, v2)
                    
                    #edges with counterclockwise higher index than ref edge
                    edges = []
                    append_next_edge(edges, central_edges[k], mesh, vi, v2)
                    for e_i in edges:            
                        #calculate edge length
                        e_len = edge_length(e_i, mesh)
                        coeff = (1)/(e_len*Nc) 
                        row_ind[j] = parent
                        col_ind[j] = fes_dual.GetDofNrs(e_i)[0]
                        data[j] = coeff * s
                        j += 1

                        append_next_edge(edges, e_i, mesh, vi, v2)
        #BND Edges ------------------------------------------------------------
        else:
            #deal with the one central edge
            e_len = edge_length(central_edges[0], mesh)
            coeff = (1)/(2*e_len) 
            row_ind[j] = parent
            col_ind[j] = fes_dual.GetDofNrs(central_edges[0])[0]
            data[j] = coeff * -1
            j += 1

            for k in range(2):
                vi = verts[k]       
                s = sign[k]     
                Nc = len(vi.faces)/2
            
                #deal with reference edge (part of original edge)
                ref_edge = get_ref_edge(central_edges[0], mesh, vi, v2)
                e_len = edge_length(ref_edge, mesh)
                coeff = (2 - Nc)/(2*e_len*Nc) 
                row_ind[j] = parent
                col_ind[j] = fes_dual.GetDofNrs(ref_edge)[0]
                data[j] = coeff * s
                j += 1

                numerator = [(1-Nc), (1)]
                edges = []
                append_next_edge(edges, central_edges[0], mesh, vi, v2)
                for e_i in edges:            
                    #calculate edge length
                    e_len = edge_length(e_i, mesh)
                    coeff = numerator[k]/(e_len*Nc) 
                    row_ind[j] = parent
                    col_ind[j] = fes_dual.GetDofNrs(e_i)[0]
                    data[j] = coeff * s 
                    j += 1

                    append_next_edge(edges, e_i, mesh, vi, v2)
        
    dual_map = sp.sparse.csr_matrix((data,(row_ind,col_ind)))#la.SparseMatrixdouble.CreateFromCOO(row_ind,col_ind,data, ne,mesh.nedge - ne)
    #print(dual_map.shape)

    # mapping to coarse primal ---------------------------------------------------------------------------------------------------------

    data_length = 24 * ne

    data = np.zeros(data_length)
    row_ind = np.zeros(data_length, dtype=int)
    col_ind = np.zeros(data_length, dtype=int)

    j = 0

    for e in pe:
        parent = e.nr  
        #print(parent)
        v0 = e.vertices[0]  
        v1 = e.vertices[1] 
        v0 = mesh.__getitem__(v0)
        v1 = mesh.__getitem__(v1)
        verts = [v0,v1]
        coeffs = [-1,1]

        i = 1
        for vi in verts:
            for el in vi.elements:
                if mesh.ngmesh.MacroElementNr(el.nr) == elnrs[parent,0] or mesh.ngmesh.MacroElementNr(el.nr) == elnrs[parent,1]:
                    el = mesh.__getitem__(el)
                    for child in el.edges:
                        
                        child = mesh.__getitem__(child)
                        if len(child.elements) == 2:
                            
                            par_el1 = mesh.ngmesh.MacroElementNr(child.elements[0].nr)
                            par_el2 = mesh.ngmesh.MacroElementNr(child.elements[1].nr)

                            #same parent element
                            if  par_el1 == par_el2:
                                row_ind[j] = parent
                                col_ind[j] = fes_primal.GetDofNrs(child)[0]
                                data[j] = coeffs[i] * 1/6
                                j += 1
                            #different parent, but both are elements of parent edge (the children of the original edge fulfill this)
                            elif par_el1 != par_el2 and (par_el1 == elnrs[parent,0] or par_el1 == elnrs[parent,1]) and (par_el2 == elnrs[parent,0] or par_el2 == elnrs[parent,1]):
                                row_ind[j] = parent
                                col_ind[j] = fes_primal.GetDofNrs(child)[0]
                                data[j] = coeffs[i] * 3/12
                                j += 1
                        #boundry
                        elif len(child.elements) == 1:
                            #the edge midpoint always has higher number than the coarse vertices, s it will always be at index 1
                            v2 = child.vertices[1]
                            v2 = mesh.__getitem__(v2)
                            #the edges which correspond to the original edge in case it was on a boundry,
                            #check if this vertex shares an edge with the other vertex of the original edge
                            #this excludes boundry edges that arent part of the original edge
                            otherv = verts[i]
                            if(set(v2.edges).intersection(otherv.edges) != set()):
                                row_ind[j] = parent
                                col_ind[j] = fes_primal.GetDofNrs(child)[0]
                                data[j] = coeffs[i] * 3/6
                                j += 1 
            i -= 1     

    primal_map = sp.sparse.csr_matrix((data,(row_ind,col_ind)))#la.SparseMatrixdouble.CreateFromCOO(row_ind,col_ind,data, ne,mesh.nedge - ne)
    #print(primal_map.shape)

    dual_L2IP = dual_map @ mat @ primal_map.T

    coo = dual_L2IP.tocoo(copy=True)

    NG_dual = la.SparseMatrixdouble.CreateFromCOO(coo.row,coo.col,coo.data, ne, ne)

    return NG_dual