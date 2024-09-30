import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, S2gate, ThermalLossChannel, LossChannel
import numpy as np
import matplotlib.pyplot as plt

def symplectic_eigenvalues(cov_matrix, hbar=2):
    n = cov_matrix.shape[0] // 2
    Omega = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    # Omega = np.kron(np.eye(n), np.array([[0, 1], [-1, 0]]))
    symplectic_matrix = np.dot(Omega, cov_matrix)
    eigvals = np.linalg.eigvals(symplectic_matrix)
    # symplectic_eigenvals = np.sort(np.abs(eigvals))[:n]
    symplectic_eigenvals = [abs(eigval) / hbar for eigval in eigvals]
    return symplectic_eigenvals 

def convert_covariance_matrix(cov_matrix, to_convention):
    if to_convention not in ['xxpp', 'xp']:
        raise ValueError("to_convention must be either 'xxpp' or 'xp'")
    n = cov_matrix.shape[0] // 2
    if to_convention == 'xp':
        # Convert from 'xxpp' to 'xp'
        indices = np.array([i // 2 + (i % 2) * n for i in range(2 * n)])
    else:
        # Convert from 'xp' to 'xxpp'
        indices = np.array([i * 2 if i < n else (i - n) * 2 + 1 for i in range(2 * n)])
    
    return cov_matrix[np.ix_(indices, indices)]

def partition_covariance_matrix(cov_matrix, indices):
    cov_matrix = np.array(cov_matrix)
    if not all(0 <= idx < cov_matrix.shape[0] for idx in indices):
        raise ValueError("Indices are out of bounds of the covariance matrix dimensions.")
    partitioned_matrix = cov_matrix[np.ix_(indices, indices)]
    
    return partitioned_matrix

def f(x):
    return (x + .5) * np.log(x + .5) - (x - .5) * np.log(x - .5)

def compute_S(cov, idx, err):
    S = 0
    xxpp_cov = convert_covariance_matrix(cov, 'xp')
    xp_cov = convert_covariance_matrix(partition_covariance_matrix(xxpp_cov, idx), 'xxpp')
    sym_eigs = symplectic_eigenvalues(xp_cov)
    for eig in sym_eigs:
        if eig < .5 - err:
            raise ValueError('symplectic eigenvalues smaller than 0.5: S')
        elif .5 - err < eig < .5 + err:
            pass
        else:
            S += f(eig)
    return S

def compute_TMI(cov, num_M2=1, err=.00001):
    I2_1 = compute_S(cov, [0, 1], err) + compute_S(cov, [2, 3], err) - compute_S(cov, range(2 + 2 * num_M2), err)
    I2_2 = compute_S(cov, [0, 1], err) + compute_S(cov, range(4, 4 + 2 * num_M2), err) - compute_S(cov, [0, 1] + list(range(4, 4 + 2 * num_M2)), err)
    I2_3 = compute_S(cov, [0, 1], err) + compute_S(cov, range(2, 4 + 2 * num_M2), err) - compute_S(cov, range(4 + 2 * num_M2), err)
    return I2_1 + I2_2 - I2_3, [I2_1, I2_2, I2_3]