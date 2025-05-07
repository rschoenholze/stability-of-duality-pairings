import matplotlib.pyplot as plt
import numpy as np
import os


#write minEV tensor from computation files to a .npy file, then read that file into this one and do the plotting

dir_path = os.path.dirname(os.path.realpath(__file__))

for d in range(2,3):
    for l in range(0,d+1):
        for st1 in ['structured mesh', 'unstructured mesh']:
            for st2 in ['', 'BND']:
                print("d{d}l{l} dual mesh".format(d=d,l=l))
                pre1 = ''
                pre2 = ''
                title = ''
                desc = ''
                if st1 == 'unstructured mesh':
                    pre1 = 'un'
                    desc = 'average '

                if st2 == 'BND':
                    pre2 = '_'
                    title = ' with Dirichlet Boundary Conditions'

                minEV = np.load(dir_path + '/'+ st1 +'/d{d}l{l}_'.format(d=d,l=l) + pre1 + 'str_dual' + pre2 + st2 +'.npy')

                symbols = ['o-','h-.','*:','+-']

                if (l == 0) or (d==3 and l==1):
                    low_order = 1
                else:
                    low_order = 0

                nMW = minEV.shape[1]
                meshwidths = np.ones(nMW)
                
                #meshwidths for l == 0 starts at 0.5
                if l == 0:
                    meshwidths *= 0.5
                elif l == 1 and st2 == 'BND':
                    meshwidths *= 0.5

                for h in range(nMW-1):
                    meshwidths[h+1] = meshwidths[h]/2

                print(meshwidths)

                high_orders = minEV.shape[0]

                #min EV
                fig, ax = plt.subplots()
                plt.grid(visible=True)
                plt.title(label="Inf-Sup Constant for {b}-forms in {a}D using dual mesh".format(a=d,b=l) + ' on a ' + st1 + title, wrap=True)
                plt.rcParams['axes.formatter.min_exponent'] = 1
                plt.loglog(meshwidths,np.ones(nMW) * np.sqrt(minEV[high_orders-1, len(meshwidths)-1]) - 0.001,'--k', label="$\mathcal{O}(c)$")

                plt.xlabel(desc + 'meshwidth h')
                plt.ylabel('$c_S$')

                lowest_high_Order = low_order + 1
                highest_high_order = lowest_high_Order + high_orders
                for i in range(lowest_high_Order,highest_high_order):
                    plt.loglog(meshwidths,np.sqrt(minEV[i-lowest_high_Order,:]), symbols[i-lowest_high_Order], label="high order=%i"%i)

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
                plt.savefig(dir_path + '/plots/d{a}l{b}_dual_'.format(a=d,b=l) + pre1 + 'str_' + st2 + '.pdf')




        
