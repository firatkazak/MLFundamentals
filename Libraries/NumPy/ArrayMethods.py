import numpy as np

a = np.array([1, 2, 3])
a = np.append(arr=a, values=[7, 8, 9])
print(a)
# [1 2 3 7 8 9]

a = np.array([1, 2, 3])
a = np.append(arr=a, values=[7, 8, 9])
a = np.insert(arr=a, obj=3, values=[4, 5, 6])
print(a)
# [1 2 3 4 5 6 7 8 9]

a = np.array([[1, 2, 3], [4, 5, 6]])
a = np.delete(arr=a, obj=0, axis=1)
print(a)
# [[2 3]
#  [5 6]]
