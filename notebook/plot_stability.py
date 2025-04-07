from ngsolve import *
from netgen.occ import *
import matplotlib.pyplot as plt
import numpy as np

#TODO curl div swap
#TODO priority for files

def stability_plot(minEV, dim, diffForm, BND='',meshtype = 'str', file='', save=''):
    symbols = ['o-','h-.','*:','+-']

    if (diffForm == 0) or (dim==3 and diffForm==1):
        low_order = 1
    else:
        low_order = 0

    nMW = minEV.shape[1]
    meshwidths = np.ones(nMW)
    
    label_addon1 = ''
    label_addon2 = ''

    #meshwidths for bnd cond starts at 0.5
    if BND != '':
        meshwidths *= 0.5
        label_addon1 = ', Boundary'

    for h in range(nMW-1):
        meshwidths[h+1] = meshwidths[h]/2

    match meshtype:
        case 'str':
            label_addon2 = ', structured mesh'
        case 'unstr':
            label_addon2 = ', unstructured mesh'

    print(meshwidths)

    high_orders = minEV.shape[0]

    #min EV
    fig, ax = plt.subplots()
    plt.grid(visible=True)
    plt.title(label="d={a}, l={b}".format(a=dim,b=diffForm) + label_addon1 + label_addon2)
    plt.rcParams['axes.formatter.min_exponent'] = 1
    if diffForm == 0 or diffForm == dim:
        #TODO change Label to O(c) 
        #plt.loglog(meshwidths,np.ones(nMW) * 1/d,'--k', label="1/%i"%d)
        ...
    else:
        #plt.loglog(meshwidths,np.power(meshwidths,2),'--k', label=r'$\mathcal{O}(h^{2})$')
        ...

    plt.xlabel('meshwidth h')
    plt.ylabel('minimal Eigenvalue')

    lowest_high_Order = low_order + 1
    highest_high_order = lowest_high_Order + high_orders
    for i in range(lowest_high_Order,highest_high_order):
        plt.loglog(meshwidths,minEV[i-lowest_high_Order,:], symbols[i-lowest_high_Order], label="high order=%i"%i)

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

    print(Labels)

    plt.xticks(ticks=meshwidths, labels=Labels)
    plt.tight_layout()
    if save != '':
        plt.savefig(save.format(a=dim,b=diffForm) + '.pdf')
    plt.show()
