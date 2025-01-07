import numpy as np

array_npy = np.load(file="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/NumPy/Data/Inputs/input_myarray.npy")
array_txt = np.loadtxt(fname="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/NumPy/Data/Inputs/input_myarray.txt", delimiter=",")

np.save(file="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/NumPy/Data/Outputs/output_myarray.npy", arr=array_npy)
np.savetxt(fname="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/NumPy/Data/Outputs/output_myarray.txt", X=array_txt, delimiter=",")

array_npy = np.load(file="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/NumPy/Data/Outputs/output_myarray.npy")
array_txt = np.loadtxt(fname="C:/Users/firat/OneDrive/Belgeler/Projects/MLFundamentals/Libraries/NumPy/Data/Outputs/output_myarray.txt", delimiter=",")

print(array_npy)
# [[ 1  2  3  4  5  6]
#  [ 7  8  9 10 11 12]
#  [13 14 15 16 17 18]
#  [18 19 20 21 22 23]]

print(array_txt)
# [[ 1.  2.  3.  4.  5.  6.]
#  [ 7.  8.  9. 10. 11. 12.]
#  [13. 14. 15. 16. 17. 18.]
#  [18. 19. 20. 21. 22. 23.]]
