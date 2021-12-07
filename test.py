import numpy as np

"""
HOW THE FUCK THIS MATRIX OPERATIONS WORK
"""



a = np.random.randint(0, 10, size=(5,5,3))
#a = a.astype(np.uint8)
a = np.array([[[7, 9, 0],
        [7, 2, 4],
        [9, 0, 6],
        [5, 3, 2],
        [1, 6, 8]],

       [[9, 4, 9],
        [1, 1, 1],
        [2, 7, 8],
        [1, 4, 0],
        [0, 4, 1]],

       [[5, 1, 3],
        [5, 3, 5],
        [5, 5, 7],
        [8, 7, 4],
        [9, 4, 8]],

       [[9, 3, 0],
        [0, 3, 5],
        [2, 0, 5],
        [2, 2, 1],
        [1, 4, 0]],

       [[2, 3, 0],
        [8, 2, 7],
        [7, 9, 9],
        [4, 4, 9],
        [7, 9, 1]]])

b = np.empty((5,3))
#print(np.vstack(a))
""" b[:,0] = a[:,:,0].mean(axis=0)
b[:,1] = a[:,:,1].mean(axis=0)
b[:,2] = a[:,:,2].mean(axis=0) """


""" b[:,:]  =a[:,:,:].mean(axis=0)
print(b)
print(np.hstack(b)) """


from sklearn.preprocessing import normalize

import numpy as np

a = np.array([[  0.,   3.,   6.],
       [  9.,  12.,  15.],
       [ 18.,  21.,  24.]])

sum = a.sum()
print(a/sum)
       
print(normalize(a))

