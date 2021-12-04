import numpy as np

a = np.array([[3,2,1], [6,5,4],[9,8,7]])


x = [(0,2), (0,1)]
print(a[x[0],x[1]])

print()
print(a[0:2,0:2])
print()
print(a[[0,2],[0,1]])

