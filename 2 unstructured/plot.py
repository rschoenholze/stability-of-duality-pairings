import matplotlib.pyplot as plt
import numpy as np


#write minEV tensor from computation files to a .npy file, then read that file into this one and do the plotting

for d in range(2,4):
    for l in range(0,d+1):
        print("d{d}l{l}".format(d=d,l=l))
        
        minEV = np.load('2 unstructured/d{d}l{l}_minEV_unstr'.format(d=d,l=l) + '.npy')

        symbols = ['o-','h-.','*:','+-']

        if (l == 0) or (d==3 and l==1):
            lowest_low_order = 1
        else:
            lowest_low_order = 0

        nMW = minEV.shape[2]
        meshwidths = np.ones(nMW)
        for h in range(nMW-1):
            meshwidths[h+1] = meshwidths[h]/2
        print(meshwidths)
        highest_low_order = lowest_low_order + minEV.shape[0]

        high_orders = minEV.shape[1]

        #min EV
        for j in range(lowest_low_order, highest_low_order):
            fig, ax = plt.subplots()
            plt.grid(visible=True)
            plt.title(label="d={a}, l={b}, low order={c}".format(a=d,b=l,c=j))
            plt.rcParams['axes.formatter.min_exponent'] = 1
            if l == 0 or l == d:
                #TODO change Label to O(c) 
                plt.loglog(meshwidths,np.ones(nMW) * 1/d,'--k', label="$\mathcal{O}(c)$")
                ...
            else:
                plt.loglog(meshwidths,np.power(meshwidths,2),'--k', label=r'$\mathcal{O}(h^{2})$')
                ...

            plt.xlabel('meshwidth h')
            plt.ylabel('minimal Eigenvalue')

            lowest_high_Order = j + 1
            highest_high_order = lowest_high_Order + high_orders
            for i in range(lowest_high_Order,highest_high_order):
                plt.loglog(meshwidths,minEV[j-lowest_low_order,i-lowest_high_Order,:], symbols[i-lowest_high_Order], label="high order=%i"%i)

            plt.legend()
            Labels = []
            exp = int(0)
            starting_mw = 1
            while starting_mw > meshwidths[0]:
                exp -= 1
                starting_mw /= 2

            for i in range(len(meshwidths)):
                Labels.append('$2^{%i}$'%exp)
                exp -= 1

            print(Labels)

            plt.xticks(ticks=meshwidths, labels=Labels)
            plt.tight_layout()
            #plt.show()
            plt.savefig("2 unstructured/plots/d{a}l{b}_minEv_unstr.pdf".format(a=d,b=l))

        
        #convergence rate for minimal EV
        # if l == 0:
        #     #reference value = value on finest mesh of same order
        #     for j in range(lowest_low_order, highest_low_order):
        #         fig, ax = plt.subplots()
        #         plt.grid(visible=True)
        #         plt.xlabel('meshwidth h')
        #         plt.ylabel(r"$ (\lambda - \lambda_{ref}) * \lambda_{ref}^{-1} $")
        #         plt.title(r"convergence to $\lambda_{ref}=\lambda_{min,finest}$, low order=%i" %j) 
        #         plt.loglog(meshwidths,np.power(meshwidths,2),'--k', label=r'$\mathcal{O}(h^{2})$')

        #         lowest_high_Order = j + 1
        #         highest_high_order = lowest_high_Order + high_orders
        #         for i in range(lowest_high_Order,highest_high_order):
        #             ref_val = minEV[j-lowest_low_order,i-lowest_high_Order,nMW-1]
        #             plt.loglog(meshwidths,(minEV[j-lowest_low_order,i-lowest_high_Order,:] - np.ones(nMW)*ref_val)/ref_val, symbols[i-lowest_high_Order], label="high order=%i"%i)

        #         plt.legend()
        #         plt.tight_layout()
        #         plt.show()
        #         #plt.savefig("d{a}l{b}_convergence_FineEV_o{c}.pdf".format(a=d,b=l,c=j))

        #     #uniform refrence value, finest on highest order
        #     for j in range(lowest_low_order, highest_low_order):
        #         fig, ax = plt.subplots()
        #         plt.grid(visible=True)
        #         plt.xlabel('meshwidth h')
        #         plt.ylabel(r"$ (\lambda - \lambda_{ref}) * \lambda_{ref}^{-1} $")
        #         plt.title(r"convergence to $\lambda_{ref}=\lambda_{min,finest,highest}$, low order=%i" %j) 
        #         plt.loglog(meshwidths,np.power(meshwidths,2),'--k', label=r'$\mathcal{O}(h^{2})$')

        #         lowest_high_Order = j + 1
        #         highest_high_order = lowest_high_Order + high_orders
        #         #using refrence value from highest order 
        #         ref_val = minEV[j-lowest_low_order,high_orders-1,nMW-1]
        #         for i in range(lowest_high_Order,highest_high_order):
        #             plt.loglog(meshwidths,(minEV[j-lowest_low_order,i-lowest_high_Order,:] - np.ones(nMW)*ref_val)/ref_val, symbols[i-lowest_high_Order], label="high order=%i"%i)

        #         plt.legend()
        #         plt.tight_layout()
        #         plt.show()
        #         #plt.savefig("d{a}l{b}_convergence_UniformFineEV_o{c}.pdf".format(a=d,b=l,c=j))

