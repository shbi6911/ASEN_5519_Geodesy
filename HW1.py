import numpy as np
import matplotlib.pyplot as plt

def legendre_poly(x: float | np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    p_array = np.zeros((n + 1, x.size))

    p_array[0, :] = 1.0
    if n == 0:
        return p_array

    p_array[1, :] = x

    for i in range(2, n + 1):
            p_array[i, :] = (((2 * i - 1) / i) * x * p_array[i-1, :] - 
                          ((i - 1) / i) * p_array[i-2, :])
    return p_array

def legendre_sum(n: int, dx: float = 1e-4, tol: float = 1e-12) -> float:

    # ensure exact endpoints to the x-range values
    N = round(2/dx) + 1
    # check agreement with input dx
    if np.abs(2/(N-1) - dx) > tol:
        raise ValueError("Input dx does not divide evenly into [-1,1] to tolerance.")
    
    x = np.linspace(-1, 1, N)

    matrix = legendre_poly(x, n)

    return np.sum(matrix[-1, :] ** 2) * 2/(N-1)

def main_1():

    # establish specified array of x-values between -1 and 1
    n_max = 1000
    dx = 1e-4
    N = round(2/dx) + 1
    x = np.linspace(-1, 1, N)

    # compute summations for all N from 0 to N_max
    matrix = legendre_poly(x, n_max)
    n_arr = np.arange(0, n_max + 1)
    s_n = np.sum(matrix**2, axis=1) * (2/(N-1))
    e_n = s_n - 2/(2*n_arr + 1)

    table_n = np.array([10, 50, 100, 500, 1000])
    for n in table_n:
        print(f"Summation is {s_n[n]:.8e} for n = {n}, 2/(2n+1) "
              f"= {2/(2*n + 1):.8e}")

    # --- plot E(n) --- *** this code is AI-written
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n_arr, e_n, lw=1.2)
    ax.set_xlabel('n')
    ax.set_ylabel(r'$E(n) = S(n) - 2/(2n+1)$')
    ax.set_title('Quadrature error vs. polynomial degree')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # plt.savefig('legendre_En.png', dpi=200)
    plt.show()
    # *** end AI-written code

def main_2():

    n_max = 15
    matrix = legendre_poly((1/2), n_max)[:, 0]
    table_n = np.array([1, 3, 5, 10, 15])
    coeff_n = 1 / 2 ** (np.arange(n_max + 1) + 1)
    for n in table_n:
        s = np.sum((coeff_n * matrix)[0:n+1])
        print(f"Summation is {s:.6f} for n = {n}, "
              f"error is {s - 1/(np.sqrt(3)):.8e}")

if __name__ == "__main__":
    main_1()
    main_2()






            
    