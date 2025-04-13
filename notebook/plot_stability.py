from ngsolve import *
from netgen.occ import *
import matplotlib.pyplot as plt
import numpy as np

def stability_plot(minEV, dim, diffForm, BND='',meshtype = 'str', file='', dual_mesh = False, swap_HC_HD=False):
    '''
    minEV is a matrix containing the minimal eigenvalues of the generalized eigenvalue problem. 

    dim needs to be 2 or 3.

    diffForm <= dim, determines the Whitney forms, Important as the lowest order isnt the same for all spaces.

    BND are the boundary conditions, if none are provided, its assumed that there are none.

    meshtype determines if the mesh is structured ('str') or unstructured (anything else).

    file, if provided saves the plot to that path, if not provided plot wont be saved.

    dual_mesh, wether the dual mesh method to approximate the dual space was used or not, Flase if not provided.
    '''
    symbols = ['o-','h-.','*:','+-']

    if dual_mesh==True and dim!=2:
        print("dual mesh is only available in 2D")
        return 

    if (diffForm == 0) or (dim==3 and diffForm==1 and swap_HC_HD==False) or (dim==3 and diffForm==2 and swap_HC_HD==True) or (dim==2 and diffForm==1 and swap_HC_HD==True):
        low_order = 1
    else:
        low_order = 0

    nMW = minEV.shape[1]
    meshwidths = np.ones(nMW)
    
    #used to automatically title plot
    label_addon1 = ''
    label_addon2 = ''
    ylabel_pre = ''
    label_addon3 = ''

    #meshwidths for bnd cond starts at 0.5
    if BND != '':
        meshwidths *= 0.5
        label_addon1 = ', Boundary'

    for h in range(nMW-1):
        meshwidths[h+1] = meshwidths[h]/2

    match meshtype:
        case 'str':
            label_addon2 = ', structured mesh'
        case _:
            label_addon2 = ', unstructured mesh'
            ylabel_pre = 'average '

    if dual_mesh == True:
        label_addon3 = ', dual mesh'

    high_orders = minEV.shape[0]

    #min EV
    fig, ax = plt.subplots()
    plt.grid(visible=True)
    plt.title(label="Inf-Sup Constant for {b}-forms in {a}D".format(a=dim,b=diffForm) + label_addon1 + label_addon2 + label_addon3)
    plt.rcParams['axes.formatter.min_exponent'] = 1
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='minor',     # minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    if diffForm == 0 or diffForm == dim or dual_mesh==True: 
        plt.loglog(meshwidths,np.ones(nMW) * np.sqrt(minEV[high_orders-1, len(meshwidths)-1]) - 0.001,'--k', label="$\mathcal{O}(c)$")
        ...
    else:
        plt.loglog(meshwidths,np.power(meshwidths,1),'--k', label=r'$\mathcal{O}(h)$')
        ...

    plt.xlabel(ylabel_pre + 'meshwidth h')
    plt.ylabel('$c_S$')

    lowest_high_Order = low_order + 1
    highest_high_order = lowest_high_Order + high_orders
    for i in range(lowest_high_Order,highest_high_order):
        plt.loglog(meshwidths,np.sqrt(minEV[i-lowest_high_Order,:]), symbols[i-lowest_high_Order], label="high order=%i"%i)

    plt.legend()
    Labels = []
    exp = int(0)
    starting_mw = 1
    while starting_mw > meshwidths[0] + 0.00001:
        exp -= 1
        starting_mw /= 2

    for i in range(len(meshwidths)):
        Labels.append('$2^{%i}$'%exp)
        exp -= 1

    plt.xticks(ticks=meshwidths, labels=Labels)
    plt.tight_layout()
    if file != '':
        plt.savefig(file + '.pdf')
    else:
        plt.show()
