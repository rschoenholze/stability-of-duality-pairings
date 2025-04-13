from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

#TODO check  that new datalength works
def H1_Primal_Dual_L2IP(mesh, swap_HC_HD=False):
    '''Returns a NGSolve sparse Matrix that corresponds to the L2 inner Produxt, between lagrangian 
    piecewise linears and the dual space constructed on piecewise constant basis functions.
    
    based on the paper:
        "OPTIMAL OPERATOR PRECONDITIONING FOR GALERKIN BOUNDARY ELEMENT METHODS ON 3-DIMENSIONAL SCREENS"
        by RALF HIPTMAIR , CARLOS JEREZ-HANCKES , AND CAROLINA URZÚA-TORRE

    The mesh is changed during execution.'''

    #iterable of vertices in original mesh, the parent vertices
    pv = mesh.vertices
    #number vertices in original mesh, sets matrix dimesion
    nv = len(pv)

    #Barycentric Refinement, 
    mesh.ngmesh.SplitPowellSabin()

    #Gives information about the primal space to map the refined dofs to the coarse ones 
    #H1(...) is the space of lagrangian piecewise linears, Compress removes unused Dofs from the refinement
    fes_primal = Compress(H1(mesh, order=1))

    #Gives informtaion about the space used to construct the weighting matrices for the dual space
    #L2(...) is the space of piecewise constants
    fes_dual = Compress(L2(mesh, order=0))

    #used in BiliniarForm
    H1_u = fes_primal.TrialFunction()
    L2_v = fes_dual.TestFunction()

    #L2 inner product of H1 and L2 basis function on barycentrically refined mesh
    H1_L2_L2IP = BilinearForm(trialspace=fes_primal, testspace=fes_dual) 
    H1_L2_L2IP += H1_u * L2_v * dx

    #create matrix corresponding to above inner product
    H1_L2_L2IP.Assemble()

    #transform NGSolve matrix to sparse scipy matrix
    rows,cols,vals  = H1_L2_L2IP.mat.COO()
    mat = sp.sparse.csr_matrix((vals,(rows,cols)))

    ########################### Weight matrix for dual space ###########################
    #each Dof is associated with a vertex in the coarse mesh, and its basis function is the 
    #linear combination of faces (2D elements) containing the vertex

    #reserve space for sparse scipy matrix construction
    data_length = mesh.ne
    # for v in pv:
    #     data_length += len(v.faces)
    data1 = np.zeros(data_length)
    row_ind1 = np.zeros(data_length, dtype=int)
    col_ind1 = np.zeros(data_length, dtype=int)

    #index, increased for every non zero entry
    j = 0

    #loop through the parent vertices
    for v in pv:
        #the coefficent in the linear combination
        coeff =  1

        #loop through all refined elements containing v, in this function space each element is associated with one Dof
        for el in v.elements:
            #get DofNr of refined element
            el_dofNr = fes_dual.GetDofNrs(el)
            #row index is the refined elemensts DofNr
            row_ind1[j] = el_dofNr[0]
            #col index is the parent elemnts DofNr, which corresponds to the vertex number the DOF is associated with
            col_ind1[j] = v.nr
            #the matrix entry is the coefficient
            data1[j] = coeff
            j += 1
    
    #construct sparse scipy matrix
    dual_map = sp.sparse.csr_matrix((data1,(row_ind1,col_ind1)))
    
    ########################### Weight matrix for coarse primal space ###########################

    #reserve space for sparse matrix
    data_length = 0
    for par in pv:
        for el in par.elements:
            #each refined element that contains the parent vertex, 
            #contains 3 vertices that are necessary to reconstruct the coarse basis function
            data_length += 3

    data = np.zeros(data_length)
    row_ind = np.zeros(data_length, dtype=int)
    col_ind = np.zeros(data_length, dtype=int)

    #index, increased for every non zero entry
    j = 0

    for par in pv:
        #Dofs associated with parent vertices keep the same Dof number after refinement
        par_dofnr = fes_primal.GetDofNrs(par)[0]
        #number of refined faces containing the parent vertex, same as number of refined elements containing parent vertex 
        par_nf = len(par.faces)
        #loop through elements containing parent
        for el in par.elements:
            #needed to access mesh information
            ele = mesh.__getitem__(el)
            #loop through vertices in el
            for v in ele.vertices:
                #needed to acces mesh information
                ver = mesh.__getitem__(v)
                #DofNr associated with child vertex is the column index
                child_dofnr = fes_primal.GetDofNrs(v)[0]
                col_ind[j] = child_dofnr
                #DofNr associated with parent vertex is the row index
                row_ind[j] = par_dofnr
                #the parent vertex itself will be checked par_nf times, the entry will add up to 1
                if par_dofnr == child_dofnr:
                    data[j] = 1/par_nf
                #vertices in the barycenters, need a coeff of 1/3, but they are checked twice
                elif len(ver.faces) == 6:
                    data[j] = 1/6
                #edge midpoints that lie on the boundary
                elif len(ver.faces) == 2:
                    data[j] = 1/2
                #edge midpoints that are not on the boundary are checked twice
                else:
                    data[j] = 1/4
                j += 1

    #construct sparse scipy matrix
    primal_map = sp.sparse.csr_matrix((data,(row_ind,col_ind)))

    #L2 inner Product of the coarse piecewise linears and the dual basis
    #Doing the multiplication in scipy makes it possible to use NGSolves .Inverse() 
    dual_L2IP =  dual_map.T @ mat @ primal_map.T 

    #needs to be in scipy's coo format, to convert to a NGSolve matrix
    coo = dual_L2IP.tocoo(copy=True)

    #Convert to NGSolve matrix
    NG_dual = la.SparseMatrixdouble.CreateFromCOO(coo.row,coo.col,coo.data, nv, nv)

    return NG_dual


#TODO check that new data length works
def L2_Primal_Dual_L2IP(mesh, swap_HC_HD=False):
    '''Returns a NGSolve sparse Matrix that corresponds to the L2 inner Produxt, between 
    piecewise constants and the dual space constructed on piecewise linear basis functions.

    based on the paper:
        "OPTIMAL OPERATOR PRECONDITIONING FOR GALERKIN BOUNDARY ELEMENT METHODS ON 3-DIMENSIONAL SCREENS"
        by RALF HIPTMAIR , CARLOS JEREZ-HANCKES , AND CAROLINA URZÚA-TORRES
    
    The mesh is changed during execution.'''

    #Barycentric Refinement
    mesh.ngmesh.SplitPowellSabin()

    #for fine to coarse mapping
    fes_primal = Compress(L2(mesh, order=0, dirichlet="bottom|right|top|left"))

    #for mapping to dual
    fes_dual = Compress(H1(mesh, order=1, dirichlet="bottom|right|top|left"))

    #needed in BiliniarForm
    L2_u = fes_primal.TrialFunction()
    H1_v = fes_dual.TestFunction()

    #L2 inner product of H1 and L2 basis function on barycentrically refined mesh
    L2_H1_L2IP = BilinearForm(trialspace=fes_primal, testspace=fes_dual) 
    L2_H1_L2IP += L2_u * H1_v * dx

    #create matrix corresponding to above inner product
    L2_H1_L2IP.Assemble()

    #transform NGSolve matrix to sparse scipy matrix
    rows,cols,vals  = L2_H1_L2IP.mat.COO()
    mat = sp.sparse.csr_matrix((vals,(rows,cols)))

    ########################### weight matrix for dual space ###########################
    #each refined element has 3 vertices
    data_length = mesh.ne * 3
    data = np.zeros(data_length)
    row_ind = np.zeros(data_length, dtype=int)
    col_ind = np.zeros(data_length, dtype=int)

    #index, increased for every non zero entry
    j = 0

    #mesh.Elements() returns a iterable of the elements
    #a element (in 2D) is associated with one face, 3 edges, 3 vertices
    for el in mesh.Elements():
        #number and DofNr of parent element
        parent = mesh.ngmesh.MacroElementNr(el.nr)
        for v in el.vertices:
            #needed to get mesh information
            ver = mesh.__getitem__(v)
            #the DofNr of the refined vertex in the H1 space
            dofnr = fes_dual.GetDofNrs(ver)
            #row index is DofNr of parent element
            row_ind[j] = parent
            #col index is refined vertex DofNr
            col_ind[j] = dofnr[0]
            #The actuel coefficient for a vertex, is 1 diveded by the number of coarse Triangles it is in,
            #equal to 2 divided by the number of refined triangles it is in,
            #since we go through all elements this will add up to the correct value
            data[j] = 1/len(ver.elements)
            j += 1

    #construct sparse scipy matrix
    dual_map = sp.sparse.csr_matrix((data,(row_ind,col_ind)))

    ########################### mapping to coarse primal ###########################
    #each refined element needs to be associated to its parent element
    data_length = mesh.ne
    data = np.zeros(data_length)
    row_ind = np.zeros(data_length, dtype=int)
    col_ind = np.zeros(data_length, dtype=int)

    #index, increased for every non zero entry
    j = 0

    #mesh.Elements() returns a iterable of the elements
    #a element (in 2D) is associated with one face, 3 edges, 3 vertices
    for el in mesh.Elements():
        #element and DofNr of Parent element 
        parent = mesh.ngmesh.MacroElementNr(el.nr)
        for dofnr in fes_primal.GetDofNrs(el):
            #row index is DofNr of parent element 
            row_ind[j] = parent
            #col index is DofNr of child element
            col_ind[j] = dofnr
            #the entries are all just 1, since this is the space of piecewise constants
            data[j] = 1
            j += 1

    #construct sparse scipy matrix
    primal_map = sp.sparse.csr_matrix((data,(row_ind,col_ind)))
    
    #L2 inner Product of the coarse piecewise consants and the dual basis
    #Doing the multiplication in scipy makes it possible to use NGSolves .Inverse() 
    dual_L2IP = dual_map @ mat @ primal_map.T

    #needs to be in scipy's coo format, to convert to a NGSolve matrix
    coo = dual_L2IP.tocoo(copy=True)

    #Convert to NGSolve matrix
    NG_dual = la.SparseMatrixdouble.CreateFromCOO(coo.row,coo.col,coo.data, mesh.ne//6, mesh.ne//6)

    return NG_dual

#############################################################################
#Helper functions for inner product on the dual mesh for differential 1 forms
def edge_length(e, mesh):
    '''returns the length of an NGSolve edge
    
    e is the edge.
    
    mesh is the mesh the edge is in.'''
    p1 = mesh.__getitem__(e.vertices[1]).point
    p0 = mesh.__getitem__(e.vertices[0]).point
    e_len = sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
    return e_len

def append_next_edge(edges, prev_edge, mesh, vi, v2):
    '''appends the next edge rotating around vi
    
    edges is the list to which next edge is appended

    prev_edge is the edge from which the nect one is searched
    
    mesh is the mesh containing the edge
    
    vi must be contained in the edge to be viable next edge
    
    v2 must not be contained in edge, this is to exclude reference edged (edges that are part of the parent edge)'''
    for el in prev_edge.elements:
        el = mesh.__getitem__(el)
        for next_e in el.edges:
            next_e = mesh.__getitem__(next_e)
            #check that the edge is not already in edges, vi must be a vertex of the edge and v2 cant be a vertex of the edge 
            if (next_e not in edges) and (vi in next_e.vertices) and (v2 not in next_e.vertices):
                edges.append(next_e)
                return
            
def get_ref_edge(central_edge, mesh, vi, v2):
    '''returns one reference edge
    
    centra_edge is one of the dual edges of the refined mesh

    mesh is the mesch containing the edge

    vi is the vertex of the parent edge that must be contained in the reference edge

    v2 is the vertex in the edge midpoint

    '''
    for el in central_edge.elements:
        el = mesh.__getitem__(el)
        for next_e in el.edges:
            next_e = mesh.__getitem__(next_e)
            #check that the relevant edges are contained
            if (vi in next_e.vertices) and (v2 in next_e.vertices):
                return next_e
###############################################################################

def HDiv_Primal_Dual_L2IP(mesh, swap_HC_HD=False):
    '''Returns a NGSolve sparse Matrix that corresponds to the L2 inner Produxt, between 
    edge conforming Raviert Thomas (subspace of H(div)) and first kind Nédélec elements (subspace of H(curl)).
    
    Which elements form the primal and which the dual elements, depends on swap_HC_HD:
    False: Raviert Thomas elements are the primal elements and 
        the dual basis is constructed as a linear combination of the first kind Nédélec elements.
    True: first kind Nédélec elements are the primal elements and dual elements are constructed as 
        linear combinations of Raviert Thomas Elements
    
    Based on the paper:
        "A Multiplicative Calderon Preconditioner for theElectric Field Integral Equation" 
        by Francesco P. Andriulli, Kristof Cools, Hakan Ba˘gci, Femke Olyslager, Annalisa Buffa,
        Snorre Christiansen, and Eric Michielssen        

    The mesh is changed during execution.
    
    The weight matrix for the dual basis loops through python lists, which is very slow'''

    #iterable of edges in the coarse mesh, the parent edges
    pe = mesh.edges
    #number of edges in the coarse mesh
    ne = mesh.nedge

    #allows to differntiate boundary edges and interior edges based on if the entries are different
    elnrs = np.ones((ne,2),dtype=int)
    for e in mesh.edges:
        #needed to access mesh information
        el1 = mesh.__getitem__(e.elements[0])
        #set both entries to the element containing edge with the lower element number
        elnrs[e.nr,:] = el1.nr
        if len(e.elements) == 2:
            el2 = mesh.__getitem__(e.elements[1])
            #if two different elements contain the edge, it is an interior edge,
            #set the second entry to the second elements number
            elnrs[e.nr,1] = el2.nr

    #Barycentric Refinement
    mesh.ngmesh.SplitPowellSabin()

    #set correct function space, 
    #depending on if first kind Nédélec elements or Raviert Thomas elements should be primal elements
    if swap_HC_HD == True:
        #for fine to coarse mapping
        fes_primal = Compress(HCurl(mesh, order=1, type1=True))

        #for mapping to dual
        fes_dual =  Compress(HDiv(mesh, order=0, RT=True))
    else:
        #for mapping to dual
        fes_dual = Compress(HCurl(mesh, order=1, type1=True))

        #for fine to coarse mapping
        fes_primal =  Compress(HDiv(mesh, order=0, RT=True))

    #needed for bilinear Form
    Hprimal_u = fes_primal.TrialFunction()
    Hdual_v = fes_dual.TestFunction()

    #L2 inner product of H1 and L2 basis function on barycentrically refined mesh
    HD_HC_L2IP = BilinearForm(trialspace=fes_primal, testspace=fes_dual) 
    HD_HC_L2IP += Hprimal_u * Hdual_v * dx

    #create matrix corresponding to above inner product
    HD_HC_L2IP.Assemble()

    #transform NGSolve matrix to sparse scipy matrix
    rows,cols,vals  = HD_HC_L2IP.mat.COO()
    mat = sp.sparse.csr_matrix((vals,(rows,cols)))

    ########################### mapping to dual ###########################
    #for each parent edge, 
    #a coefficient is assigned to the refined edges containing the vertices of the parent edge,
    #additionally the dual edges (edges from the midpoint of the parent edge to the barycenters) also need a coefficent.
    #the new refined edges that were part of the parent edge only carry a coefficient for edges that have vertex on the boundary
    data_length = 0
    #iterate through parent edges
    for e in pe:
        #iterate through vertices in the edge e
        for v in e.vertices:
            #needed to get mesh information
            v = mesh.__getitem__(v)
            #the plus 4 is technically too high, but it doesnt matter as all these entries are just 0 with index (0,0)
            data_length += len(v.edges) + 4

    data = np.zeros(data_length)
    row_ind = np.zeros(data_length, dtype=int)
    col_ind = np.zeros(data_length, dtype=int)

    #index, increased for every non zero entry
    j = 0

    for e in pe:
        #the edge number in the coarse mesh corresponds to the Dof number in the lowest order space 
        parent = e.nr  
        #getting acces to the mesh information of the vertices of the parent edge
        v0 = e.vertices[0]  
        v1 = e.vertices[1] 
        v0 = mesh.__getitem__(v0)
        v1 = mesh.__getitem__(v1)

        #bnd edges need to be handled differently
        bnd_edge = False
        if elnrs[parent, 0] == elnrs[parent,1]:
            bnd_edge = True

        #finding the dual edges of the parent edge
        central_edges = []
 
        #dual edges must be in elements that contain v0, v1
        #the edges that these elements share are the dual edges 
        for el0 in v0.elements:
            for el1 in v1.elements:
                el0 = mesh.__getitem__(el0)
                el1 = mesh.__getitem__(el1)
                #check if the elements have an edge together
                intersect = tuple(set(el0.edges).intersection(el1.edges))
                if intersect != ():
                    e = mesh.__getitem__(intersect[0])
                    central_edges.append(e)

        #finding vertex in edge midpoint
        #for interior edges, the vertex contained in both dual edges is the edge modpoint.
        if len(central_edges) == 2:
            v2 = tuple(set(central_edges[0].vertices).intersection(central_edges[1].vertices))[0]
            v2 = mesh.__getitem__(v2)
        #For boundary edges, there is only one dual edge, with two vertices, one is a barycenter and thus has 6 vaces containing it
        #the other is on the boundary, has 2 faces containing it and is the edge midpoint
        else:
            for v in central_edges[0].vertices:
                v = mesh.__getitem__(v)
                if len(v.faces) == 2:
                    v2 = v

        #setting the coefficients for the edges
        verts = [v0,v1]
        #since the dual elements are oriented from one vertex to the other,
        #one of the vertices needs all coefficents assigned to its edges be multiplied by -1
        sign = [1, -1]
        #deal with boundary edges seperatly
        if bnd_edge == False:
            #k will also index vertex and sign lists
            for k in range(2):
                vi = verts[k]
                #Nc is the number of coarse faces the vertex is in
                #which is equal to the number of refined faces dived by 2, it is in
                Nc = len(vi.faces)/2

                #vertices that are on the boundary have different coefficents 
                bnd_vertex = False
                for bnd_el in mesh.Elements(BND):
                    if vi in bnd_el.vertices:
                        bnd_vertex = True

                #sign for current vertex
                s = sign[k]
                #initialize edges list
                edges = []

                #INTERNAL vertex ---------------------------------------------------------------------
                if bnd_vertex == False:
                    #each vertex is associated with one central edge
                    edges = [central_edges[k]]
                    
                    #the coefficents depend on where the edge is with respect to the dual edge,
                    #i increases with rotation around vi
                    i = 0    
                    for e_i in edges:            
                        
                        #calculate edge length
                        e_len = edge_length(e_i, mesh)
                        coeff = (Nc - i)/(2*e_len*Nc) 
                        #row index is DofNr of parent edge
                        row_ind[j] = parent
                        #col index is DofNr of child edge
                        col_ind[j] = fes_dual.GetDofNrs(e_i)[0]
                        #entry is coeff multiplied by sign for this vertex
                        data[j] = coeff *s
                        j += 1

                        #find next edge rotated around vi, containg vi,
                        #reference edges (which are part of the original edge) dont have a coefficient for internal vertices.
                        append_next_edge(edges, e_i, mesh, vi, v2)
                        i += 1
                else:
                    #BND vertex (of a non Boundary Edge)--------------------------------------------------------------------------
                    #deal with reference edge (part of original edge)
                    ref_edge = get_ref_edge(central_edges[0], mesh, vi, v2)
                    e_len = edge_length(ref_edge, mesh)
                    coeff = (2 - Nc)/(2*e_len*Nc) 
                    #row index is DofNr of parent edge
                    row_ind[j] = parent
                    #col index is DofNr of child edge
                    col_ind[j] = fes_dual.GetDofNrs(ref_edge)[0]
                    #entry is coeff multiplied by sign for this vertex
                    data[j] = coeff *s
                    j += 1

                    #deal with dual edge
                    e_len = edge_length(central_edges[k], mesh)
                    coeff = (1)/(2*e_len) 
                    #row index is DofNr of parent edge
                    row_ind[j] = parent
                    #col index is DofNr of child edge
                    col_ind[j] = fes_dual.GetDofNrs(central_edges[k])[0]
                    #entry is coeff multiplied by sign for this vertex
                    data[j] = coeff *s
                    j += 1

                    #edges have different coefficients based on if their index 
                    #is lower or higher than the reference edge, the paper uses counterclockwise indexing
                    #so that is used to differentiate it, but technically the implementation doesnt follow that

                    #edges with counterclockwise lower index than ref edge
                    append_next_edge(edges, central_edges[k-1], mesh, vi, v2)
                    for e_i in edges:            
                        #calculate edge length
                        e_len = edge_length(e_i, mesh)
                        coeff = (1-Nc)/(e_len*Nc) 
                        #row index is DofNr of parent edge
                        row_ind[j] = parent
                        #col index is DofNr of child edge
                        col_ind[j] = fes_dual.GetDofNrs(e_i)[0]
                        #entry is coeff multiplied by sign for this vertex
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
                        #row index is DofNr of parent edge
                        row_ind[j] = parent
                        #col index is DofNr of child edge
                        col_ind[j] = fes_dual.GetDofNrs(e_i)[0]
                        #entry is coeff multiplied by sign for this vertex
                        data[j] = coeff * s
                        j += 1

                        append_next_edge(edges, e_i, mesh, vi, v2)
        #BND Edges ------------------------------------------------------------
        else:
            #deal with the one central edge
            e_len = edge_length(central_edges[0], mesh)
            coeff = (1)/(2*e_len) 
            #row index is DofNr of parent edge
            row_ind[j] = parent
            #col index is DofNr of child edge
            col_ind[j] = fes_dual.GetDofNrs(central_edges[0])[0]
            #since there is only 1 dual edge, use same sign for all boundary edges
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
                #row index is DofNr of parent edge
                row_ind[j] = parent
                #col index is DofNr of child edge
                col_ind[j] = fes_dual.GetDofNrs(ref_edge)[0]
                #entry is coeff multiplied by sign for this vertex
                data[j] = coeff * s
                j += 1

                #numerators foe the coefficients
                numerator = [(1-Nc), (1)]
                edges = []
                append_next_edge(edges, central_edges[0], mesh, vi, v2)
                for e_i in edges:            
                    #calculate edge length
                    e_len = edge_length(e_i, mesh)
                    coeff = numerator[k]/(e_len*Nc) 
                    #row index is DofNr of parent edge
                    row_ind[j] = parent
                    #col index is DofNr of child edge
                    col_ind[j] = fes_dual.GetDofNrs(e_i)[0]
                    #entry is coeff multiplied by sign for this vertex
                    data[j] = coeff * s 
                    j += 1

                    append_next_edge(edges, e_i, mesh, vi, v2)
        
    #construct sparse scipy matrix
    dual_map = sp.sparse.csr_matrix((data,(row_ind,col_ind)))

    ########################### mapping to coarse primal ###########################
    #to reconstruct the coarse basis functions only the interior child edges of the 
    #faces associated with a parent edge need a coefficient, but it is difficult to iterate 
    #only through the relevant edges, but due to the orientation of the edges being different  
    #the unnecessary edges cancel themselves out.
    data_length = 24 * ne

    data = np.zeros(data_length)
    row_ind = np.zeros(data_length, dtype=int)
    col_ind = np.zeros(data_length, dtype=int)

    #index, increased for every non zero entry
    j = 0

    #iterate through parent edges
    for e in pe:
        #edge Nr of parent vertex, corresponds to DofNr in the coarse space
        parent = e.nr  
        #the vertices of the parent edge
        v0 = e.vertices[0]  
        v1 = e.vertices[1] 
        v0 = mesh.__getitem__(v0)
        v1 = mesh.__getitem__(v1)
        verts = [v0,v1]
        #sign of the coefficients depends on which vertex the child edge belongs to
        coeffs = [-1,1]

        #indexes coeff array
        i = 1
        #loop through parent edge vertices
        for vi in verts:
            #loop through elements containg vi
            for el in vi.elements:
                #Only set coefficients 
                #if the elements parent element is one of the elements that contain the parent edge 
                if mesh.ngmesh.MacroElementNr(el.nr) == elnrs[parent,0] or mesh.ngmesh.MacroElementNr(el.nr) == elnrs[parent,1]:
                    #needed to access mesh information
                    el = mesh.__getitem__(el)
                    #loop through the edges in the element
                    for child in el.edges:
                        #needed to access mesh information
                        child = mesh.__getitem__(child)
                        if len(child.elements) == 2:
                            #get parent elements
                            par_el1 = mesh.ngmesh.MacroElementNr(child.elements[0].nr)
                            par_el2 = mesh.ngmesh.MacroElementNr(child.elements[1].nr)

                            #same parent element, thus this edge is a interior edge
                            #the dual edges (edges from the parent edge midpoint to the barycenter) are included here but dont need a coefficient,
                            #but their coefficents add up to 0 since they are checked from both v0 and v1
                            #the interior edges that are closer to the parent edge need a value of 2/6, but they are checked by 2 elements 
                            #so their coefficients add up correctly
                            if  par_el1 == par_el2:
                                row_ind[j] = parent
                                col_ind[j] = fes_primal.GetDofNrs(child)[0]
                                data[j] = coeffs[i] * 1/6
                                j += 1
                            #different parent, but both are elements of parent edge (the children of the parent edge fulfill this)
                            #the child edges that form the parent edge need coefficients of 3/6, since they are checked by 2 elements
                            #their coefficients add up correctly
                            elif par_el1 != par_el2 and (par_el1 == elnrs[parent,0] or par_el1 == elnrs[parent,1]) and (par_el2 == elnrs[parent,0] or par_el2 == elnrs[parent,1]):
                                row_ind[j] = parent
                                col_ind[j] = fes_primal.GetDofNrs(child)[0]
                                data[j] = coeffs[i] * 3/12
                                j += 1
                        #for boundry edges we only need to assign a value if the parent edge is a boundary edge
                        elif len(child.elements) == 1:
                            #get the edge midpoint
                            #since it always has a higher vertex number than the coarse vertices, it will always be at index 1
                            v2 = child.vertices[1]
                            v2 = mesh.__getitem__(v2)
                            #the edges which correspond to the parent edge in case it was on a boundry,
                            #check if this vertex shares an edge with the other vertex of the original edge
                            #this excludes boundry edges that arent part of the original edge
                            otherv = verts[i]
                            if(set(v2.edges).intersection(otherv.edges) != set()):
                                row_ind[j] = parent
                                col_ind[j] = fes_primal.GetDofNrs(child)[0]
                                data[j] = coeffs[i] * 3/6
                                j += 1 
            i -= 1     

    #construct sparse scipy matrix
    primal_map = sp.sparse.csr_matrix((data,(row_ind,col_ind)))

    #L2 inner Product of the coarse piecewise consants and the dual basis
    #Doing the multiplication in scipy makes it possible to use NGSolves .Inverse() 
    dual_L2IP = dual_map @ mat @ primal_map.T

    #needs to be in scipy's coo format, to convert to a NGSolve matrix
    coo = dual_L2IP.tocoo(copy=True)

    #Convert to NGSolve matrix
    NG_dual = la.SparseMatrixdouble.CreateFromCOO(coo.row,coo.col,coo.data, ne, ne)

    return NG_dual