import math
import torch
import pandas as pd
import numpy as np
import scipy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from pathlib import Path

class polynomial():
    def __init__(self, coefficients:list):
        self.coefficients = coefficients

    def __call__(self, x:float):
        vals = [x**i for i in range(len(self.coefficients))]
        return np.dot(vals,self.coefficients)

    def get_derivative(self):
        derivative_coeffs = [coeff*i for i, coeff in enumerate(self.coefficients)][1:]
        return polynomial(derivative_coeffs)

    def get_derivates(self, n_order:int):
        coefficients = self.coefficients
        for i in range(n_order):
            coefficients = [coeff*i for i, coeff in enumerate(coefficients)][1:]
        return polynomial(coefficients)

    @classmethod
    def regress(cls, n_order, X: np.ndarray, y:np.ndarray):
        matrix = np.empty(shape=(len(x), n_order))
        for i,x in enumerate(X):
            for j, in range(n_order):
                matrix[i,j] = x**i

        matrix_T = matrix.tranpose
        square_matrix = np.matmul(matrix_T, matrix)
        rhs = np.matmul(matrix_T, y)

        coefficients = np.linalg.solve(array, y)
        return cls(coefficients)

class general_regression_model():
    def __init__(self, functions: list):
        self.functions = functions
        self.coefficients =  np.empty(len(functions),)

    def __call__(self, X:np.ndarray):
        func_vals = [fn(X) for fn in self.functions]
        true_val = np.dot(func_vals, self.coefficients)
        return true_val

    def fit(self, X: np.ndarray, y: np.ndarray):
        matrix = np.empty(shape=(len(X), len(self.functions)))
        for i,x in enumerate(X):
            for j, fn in enumerate(self.functions):
                matrix[i,j] = fn(x)

        matrix_T = np.transpose(matrix)
        square_matrix = np.matmul(matrix_T, matrix)
        rhs = np.matmul(matrix_T, y)
        self.coefficients = np.linalg.solve(square_matrix, rhs)

    @classmethod
    def polynomial(cls, order:int):
        def get_basis_fn(i):
            def basis_fn(x):
                return x**n
            return basis_fn
        functions = [get_basis_fn(i) for i in range(order)]

        my_model = cls(functions)
        return my_model

    @classmethod
    def two_dim(cls):
        def f0(x : np.ndarray):
            return 1
        def f1(x : np.ndarray):
            return x[0]
        def f2(x : np.ndarray):
            return x[1]
        def f3(x : np.ndarray):
            return x[0] * x[1]

        my_regressions_model = cls([f1,f2,f3])
        return my_regressions_model

    @classmethod
    def two_dim_more(cls):
        def f0(x : np.ndarray):
            return 1
        def f1(x : np.ndarray):
            return x[0]
        def f2(x : np.ndarray):
            return x[1]
        def f3(x : np.ndarray):
            return x[0] * x[1]
        def f4(x : np.ndarray):
            return x[0]*x[0]
        def f5(x : np.ndarray):
            return x[1] * x[1]

        my_regressions_model = cls([f1,f2,f3])
        return my_regressions_model

#Processing calibration data and creating array with it
calibration_df = pd.read_csv('calibration_data.tsv', sep='\t')
calibration_headers = calibration_df.columns

[cp_alphas, cp_betas, cp_ts, cp_ss, alphas, betas] = [calibration_df[header].to_numpy() for header in calibration_headers]
#alpha = atan(tan(a)/cos(b))
alphas = np.array([math.atan(math.tan(math.radians(alphas[i]))/math.cos(math.radians(betas[i]))) for i in range(len(alphas))])

"""How do we find CP_5 and v_inf from the calibration dataset? How do we use/what are CP_t and CP_s?"""
cp_centers = np.zeros(len(alphas))
v_inf = np.zeros(len(alphas))
#creating whole calibration array
calibration_array = np.hstack([collumn_data.reshape(-1,1) for collumn_data in [cp_alphas, cp_betas, cp_centers, alphas, np.radians(betas), v_inf]])
print(calibration_array)
#seperating regression xs and ys
xs = calibration_array[:,:3]    #cp_alpha, cp_beta, cp_center
ys = calibration_array[:,3:]    #alphas, betas, v_inf

#STARTING REGRESSION
alpha_model = general_regression_model.two_dim_more()
beta_model = general_regression_model.two_dim_more()

alpha_model.fit(xs, ys[:,0])
beta_model.fit(xs, ys[:,1])
print(alpha_model.coefficients)
print(beta_model.coefficients)


#TESTING
alphas_true = ys[:,0]
alphas_pred = [alpha_model(x_vec) for x_vec in xs]

plt.scatter(xs[:,0], alphas_true, label="true")
plt.scatter(xs[:,0], alphas_pred, label="predict")
plt.xlabel("cp_alpha")
plt.ylabel("alpha")
plt.legend()
plt.savefig("plot2.png")
plt.show()

"""
EXAMPLE CODE
import calibration
alphas = calibration.alpha_model([cp_alphas, cp_betas])
betas = calibration.beta_model([cp_alphas, cp_betas])
"""
