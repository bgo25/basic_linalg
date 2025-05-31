# With this script, I use ctypes to call my library
import ctypes
import os
import sys

#--------------------------------------------------------------------------#
#                   LOADING THE DLL LIBRARY
#--------------------------------------------------------------------------#
try:
    vector_lib = ctypes.CDLL("./vector_operations.dll")
except OSError as e:
    print(e)
    print("DLL loading failed. Exiting.")
    sys.exit(1)


#--------------------------------------------------------------------------#
#                  ACTIVATING THE DLL FUNCTIONS
#--------------------------------------------------------------------------#

# void print_vector(double*, int);
vector_lib.print_vector.argtypes = [ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int]
vector_lib.print_vector.restype = None

# void vector_sum (double*, double*, double*, int);
vector_lib.vector_sum.argtypes = [ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double),
                                  ctypes.c_int]
vector_lib.vector_sum.restype = None

# void scalar_multiplier (double, double*, double*, int dim);
vector_lib.scalar_multiplier.argtypes = [ctypes.c_double,
                                         ctypes.POINTER(ctypes.c_double),
                                         ctypes.POINTER(ctypes.c_double),
                                         ctypes.c_int]
vector_lib.scalar_multiplier.restype = None

# double dot_product (double*, double*, int);
vector_lib.dot_product.argtypes = [ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double),
                                   ctypes.c_int]
vector_lib.dot_product.restype = ctypes.c_double

# double norm(double* v, int dim);
vector_lib.norm.argtypes = [ctypes.POINTER(ctypes.c_double),
                            ctypes.c_int]
vector_lib.norm.restype = ctypes.c_double
print("-> Functions successfully activated")


#--------------------------------------------------------------------------#
#                  WRAPPING THE C FUNCTIONS IN PURE PYTHON
#--------------------------------------------------------------------------#

def c_dot_product(list_a, list_b):
    n = len(list_a)
    # Convert the lists into 
    LocalVector = ctypes.c_double * n
    c_a = LocalVector(*list_a)
    c_b = LocalVector(*list_b)
    c_n = ctypes.c_int(n)
    res = vector_lib.dot_product(c_a, c_b, c_n)
    return res

def c_print_array(list_a):
    # Define an appropriate C vector with the given len
    n = len(list_a)
    LocalArray = ctypes.c_double * n
    c_a = LocalArray(*list_a)
    vector_lib.print_vector(c_a, n)

def c_vector_sum(v1, v2):
    n = len(v1)
    LocalArray = ctypes.c_double * n
    c1 = LocalArray(*v1)
    c2 = LocalArray(*v2)
    res = [0.] * n
    c_r = LocalArray(*res)
    vector_lib.vector_sum(c1, c2, c_r, n)
    # Now, I want to convert my LocalArray c_r in a list, back
    return list(c_r)

def c_scalar_multiplier(lam, v):
    n = len(v)
    LocalVector = ctypes.c_double * n
    c_v = LocalVector(*v)
    c_lam = ctypes.c_double(lam)
    res = [0.] * n
    c_r = LocalVector(*res)
    vector_lib.scalar_multiplier(c_lam, c_v, c_r, n)
    return list(c_r)

def c_norm(v):
    n = len(v)
    LocalVector = ctypes.c_double * n
    c_v = LocalVector(*v)
    return vector_lib.norm(c_v, n)


#--------------------------------------------------------------------------#
#                              MAIN EXECUTION
#--------------------------------------------------------------------------#
if __name__ == "__main__":
    print("Let's test some results!")
    a = [0., 1.]
    b = [1., 0.]
    val = 10.
    c_print_array(a)
    print(f"{a} + {b} = {c_vector_sum(a, b)}")
    print(f"{val} * {a} = {c_scalar_multiplier(val, a)}")
    print(f"{a} dot {b} = {c_dot_product(a, b)}")
    print(f"norm({a}) = {c_norm(a)}")
