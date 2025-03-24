import matplotlib.pyplot as plt
import numpy as np


#write minEV tensor from computation files to a .npy file, then read that file into this one and do the plotting

for d in range(2,3):
    for l in range(0,d+1):
        if l == 1:
            continue

        for st1 in ['structured mesh', 'unstructured mesh']:
            for st2 in ['', 'BND']:
                print("d{d}l{l} dual mesh".format(d=d,l=l))
                pre1 = ''
                pre2 = ''
                if st1 == 'unstructured mesh':
                    pre1 = 'un'
                if st2 == 'BND':
                    pre2 = '_'

                minEV = np.load('3 dual mesh exp/'+ st1 +'/d{d}l{l}_'.format(d=d,l=l) + pre1 + 'str_dual' + pre2 + st2 +'.npy')

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

                for h in range(nMW-1):
                    meshwidths[h+1] = meshwidths[h]/2

                print(meshwidths)

                high_orders = minEV.shape[0]

                #min EV
                fig, ax = plt.subplots()
                plt.grid(visible=True)
                plt.title(label="d={a}, l={b}, low order={c}".format(a=d,b=l,c=low_order) + ', ' + st1 + ',' + st2)
                plt.rcParams['axes.formatter.min_exponent'] = 1
                if l == 0 or l == d:
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
                plt.savefig('3 dual mesh exp/d{a}l{b}_dual_'.format(a=d,b=l) + pre1 + 'str_' + st2 + '.pdf')




        
