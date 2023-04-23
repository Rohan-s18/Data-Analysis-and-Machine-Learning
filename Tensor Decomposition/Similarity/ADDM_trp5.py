import numpy as np
import pickle
import pandas as pd
import scipy.sparse
from numpy.linalg import norm, inv
from scipy.linalg import khatri_rao
from numpy import kron
import matplotlib.pyplot as plt
from scipy.sparse import kron as kr
from scipy.sparse import csr_matrix, identity
from scipy.sparse import linalg

learning_rate = 0.000001

#Transcriptional Response Profile Side information number is 5
num_side_info = 0
alpha_a = []
for i in range(num_side_info):
    alpha_a.append(1.0)


def admm(rank, rho=0.5, eta=0.5, init_lower_bound=0.0, init_upper_bound=1.0, learning_cycles=15):
    losses = []

    with open('/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/Pickles/tensor_x.pickle', 'rb') as t_x:
        x = pickle.load(t_x)
    with open('/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/Pickles/tensor_y.pickle', 'rb') as t_y:
        y = pickle.load(t_y)

    tensor_x = np.array([x_val.to_numpy() for x_val in x.values()])
    tensor_y = np.array([y_val.to_numpy() for y_val in y.values()])

    print('shape of tensor x : ', tensor_x.shape)
    print('shape of tensor y : ', tensor_y.shape)

    num_drugs = tensor_x.shape[2]
    num_disease = tensor_x.shape[0]
    num_ddi = tensor_y.shape[0]


    U = np.random.uniform(init_lower_bound, init_upper_bound, size=(num_drugs, rank))

    D = np.random.uniform(init_lower_bound, init_upper_bound, size=(num_drugs, rank))

    V = np.random.uniform(init_lower_bound, init_upper_bound, size=(num_disease, rank))

    W = np.random.uniform(init_lower_bound, init_upper_bound, size=(num_ddi, rank))

    Sa = []
    Ci = []
    Ui = []
    Qi = []

    X1 = unfold(tensor_x, 1)
    Y1 = unfold(tensor_y, 1)

    for i in range(num_side_info):
        temp_sa = pd.read_csv('/Users/rohansingh/github_repos/Data-Analysis-and-Machine-Learning/Tensor Decomposition/Datasets/Side_Information/si_{}.csv'.format(i)).fillna(0).to_numpy()[:, 1:]
        Sa.append(temp_sa)
        Ci.append(np.random.uniform(init_lower_bound, init_upper_bound, size=(num_drugs, rank)))
        Ui.append(np.random.uniform(init_lower_bound, init_upper_bound, size=(num_drugs, rank)))
        Qi.append(calculate_Qi(Ui[i]))

    # calculate the lagrangian multiplier
    F, Y = lagrangian_multiplier(U, Sa, Ui, Ci, Qi)

    current_loss = update_optimization_val(tensor_x, tensor_y, U, D, V, W, Sa, Ci, Ui, Qi, rho, eta, F, Y)
    losses.append(current_loss)
    print('current loss: ', current_loss)

    damping_counter = 0

    while learning_cycles > 0:
        gradient_U = learning_rate * update_U(V, D, W, X1, Y1, Ui, Qi, rho, F)
        U += gradient_U
        print('U finished!')
        U[U < 0] = 0

        gradient_D = learning_rate * update_D(X1, Y1, U, W, V, F, num_drugs, num_disease, num_ddi, rank, rho)
        D += gradient_D
        print('D finished!')
        D[D < 0] = 0

        if damping_counter < 2:
            gradient_V = learning_rate * update_V(X1, U, D, num_drugs, num_disease, rank, True)
            damping_counter +=1
        else:
            gradient_V = learning_rate * update_V(X1, U, D, num_drugs, num_disease, rank, False)
        V += gradient_V
        print('V finished!')
        V[V < 0] = 0

        if damping_counter < 2:
            gradient_W = learning_rate * update_W(Y1, U, D, num_drugs, num_ddi, rank, True)
            damping_counter += 1
        else:
            gradient_W = learning_rate * update_W(Y1, U, D, num_drugs, num_ddi, rank, False)
        print('W finished')
        W += gradient_W
        W[W < 0] = 0

        for i in range(num_side_info):
            gradient_Ci = learning_rate * update_Ci(i, Sa, Ui, U, Qi, Y, eta)
            current_Ci = Ci[i]
            current_Ci += gradient_Ci
            current_Ci[current_Ci < 0] = 0
            Ci[i] = current_Ci

            gradient_Ui = learning_rate * update_Ui(i, Sa, Ci, Y, Ui, eta)
            current_Ui = Ui[i]
            current_Ui += gradient_Ui
            current_Ui[current_Ui < 0] = 0
            Ui[i] = current_Ui

            Qi[i] = calculate_Qi(Ui[i])

        F, Y = lagrangian_multiplier(U, Sa, Ui, Ci, Qi)

        current_loss = update_optimization_val(tensor_x, tensor_y, U, D, V, W, Sa, Ci, Ui, Qi, rho, eta, F, Y)
        losses.append(current_loss)
        print('current loss: ', current_loss)

        learning_cycles -= 1

    plt.figure()
    plt.plot(losses, c='blue')
    plt.title('losses vs iteration')
    plt.show()

def update_D(X1, Y1, U, W, V, F, num_drugs, num_disease, num_ddi, rank, rho):
    C3 = kr(identity(num_drugs*num_disease), csr_matrix(U)) * (kr(csr_matrix(khatri_rao(np.identity(rank), V)), identity(num_drugs)))

    C4 = kr(identity(num_drugs*num_ddi), csr_matrix(U)) * kr(csr_matrix(khatri_rao(np.identity(rank), W)), identity(num_drugs))

    b1 = csr_matrix(X1.reshape(-1))
    b2 = csr_matrix(Y1.reshape(-1))

    result_temp1 = (C3.transpose() * C3).multiply(2) + (C4.transpose() * C4).multiply(2)
    result_temp1.data += rho
    f = np.ndarray(F)
    result_temp2 = (C3.transpose() * b1.transpose()).multiply(2) \
                   + (C4.transpose() * b2.transpose()).multiply(2) \
                   + csr_matrix(f.reshape(-1)).transpose() \
                   + csr_matrix(rho * U.reshape(-1)).transpose()
    result = (linalg.inv(result_temp1) * result_temp2).toarray()
    return np.transpose(result).reshape((num_drugs, rank))


def update_V(X1, U, D, num_drugs, num_disease, rank, damping):
    C1_temp = csr_matrix(khatri_rao(np.identity(num_disease*rank), np.dot(D, kron(np.identity(rank), np.full((1, num_disease), 1)))))
    C1 = csr_matrix(kr(identity(num_drugs * num_disease), csr_matrix(U))) * C1_temp

    b1 = csr_matrix(X1.reshape(-1))

    # damping the matrix a little bit for it to be solvable
    result_temp = (C1.transpose() * C1).multiply(2)

    if damping:
        c = 1
        result_temp.setdiag(result_temp.diagonal() + c)

    result = (linalg.inv(result_temp) * ((C1.transpose() * b1.transpose()).multiply(2))).toarray()
    return np.transpose(result).reshape((num_disease, rank))


def update_W(Y1, U, D, num_drugs, num_ddi, rank, damping):
    C2_temp = csr_matrix(khatri_rao(np.identity(num_ddi*rank), np.dot(D, kron(np.identity(rank), np.full((1, num_ddi), 1)))))
    C2 = csr_matrix(kr(identity(num_drugs * num_ddi), csr_matrix(U))) * C2_temp

    b2 = csr_matrix(Y1.reshape(-1))

    # damping the matrix a little bit for it to be solvable
    result_temp = (C2.transpose() * C2).multiply(2)

    if damping:
        c = 1
        result_temp.setdiag(result_temp.diagonal() + c)

    result = (linalg.inv(result_temp) * ((C2.transpose() * b2.transpose()).multiply(2))).toarray()
    return np.transpose(result).reshape((num_ddi, rank))


def update_U(V, D, W, X1, Y1, Ui, Qi, rho, F):
    pi = khatri_rao(V, D)
    theta = khatri_rao(W, D)

    pi_prime = 2 * np.dot(np.transpose(pi), pi) \
               + 2 * np.dot(np.transpose(theta), theta)

    identity = np.identity(pi_prime.shape[0])
    pi_prime_i = 0
    for i in range(num_side_info):
        pi_prime_i += alpha_a[i] * identity
    pi_prime_i = 2 * pi_prime_i

    pi_prime = pi_prime + pi_prime_i + rho * identity

    result_i = 0
    for i in range(num_side_info):
        result_i += alpha_a[i] * np.dot(Ui[i], Qi[i])
    result_i = 2 * result_i

    result = 2 * np.dot(X1, pi) + 2 * np.dot(Y1, theta) + result_i + rho * D + np.full(D.shape, F)
    result = np.dot(result, inv(pi_prime))

    return result


def update_Ci(i, Sa, Ui, U, Qi, Y, eta):
    Ci_left_half = 2 * alpha_a[i] * np.dot(Sa[i], Ui[i])\
                   + 2 * alpha_a[i] * np.dot(U, np.transpose(Qi[i])) - Y[i]\
                   + eta*Ui[i]

    Ci_right_half = 2 * alpha_a[i] * np.dot(np.transpose(Ui[i]), Ui[i]) \
                    + 2 * alpha_a[i] * np.dot(Qi[i], np.transpose(Qi[i])) \
                    + eta

    result = np.dot(Ci_left_half, inv(Ci_right_half))
    return result


def update_Ui(i, Sa, Ci, Y, Ui, eta):
    result = (1/eta) * (2 * alpha_a[i] * np.dot(Sa[i], Ci[i]) - Y[i] + eta * Ci[i] - 2 * alpha_a[i] * np.dot(np.dot(Ci[i], np.transpose(Ui[i])), Ci[i]))
    return result


def update_optimization_val(tensor_x, tensor_y, U, D, V, W, Sa, Ci, Ui, Qi, rho, eta, F, Y):
    optimization_function_val = norm(tensor_x - resemble_matrix(U, D, V)) \
                                + norm(tensor_y - resemble_matrix(U, D, W)) \
                                + side_info_opt_func_val(Sa, Ci, Ui, Qi, U)

    # calculate the result of lagrangian augmentation
    '''
    lagrangian_augmentation = 0
    lagrangian_augmentation += np.dot(np.transpose(F), D-U).trace()
    lagrangian_augmentation += (rho/2) * norm(D-U)
    for i in range(num_side_info):
        temp = np.dot(np.transpose(Y[i]), Ci[i] - Ui[i]).trace()
        lagrangian_augmentation += temp + (eta/2) * norm(Ci[i] - Ui[i])
    optimization_function_val += lagrangian_augmentation
    '''

    return optimization_function_val


def lagrangian_multiplier(U, Sa, Ui, Ci, Qi):
    F_temp = 0
    Y = []
    for i in range(num_side_info):
        temp = alpha_a[i] \
               * (-2 * np.dot(Ci[i], Qi[i]) + 2 * U)
        F_temp += temp
        temp_y = -4 * np.dot(Sa[i], Ci[i]) \
                 + 4 * np.dot(Ci[i], np.dot(np.transpose(Ui[i]), Ci[i])) \
                 + 2 * alpha_a[i] * np.dot(Ci[i], np.dot(Qi[i], np.transpose(Qi[i]))) \
                 - 2 * alpha_a[i] * np.dot(U, np.transpose(Qi[i]))
        Y.append(np.full(Ui[i].shape, temp_y))

    return F_temp, Y


def side_info_opt_func_val(Sa, Ci, Ui, Qi, U):
    result = 0.0
    for i in range(len(Sa)):
        temp = alpha_a[i] * (norm(Sa[i] - np.dot(Ci[i], np.transpose(Ui[i]))) + norm(np.dot(Ci[i], Qi[i]) - U))
        result += temp

    return result


def resemble_matrix(U, D, V):
    result = np.zeros((V.shape[0], U.shape[0], U.shape[0]), dtype=float)
    num_col = U.shape[1]
    for i in range(num_col):
        Ui = U[:, i]
        Di = D[:, i]
        Vi = V[:, i]
        result = result + three_way_outer_product(Ui, Di, Vi)

    return result


def three_way_outer_product(a, b, c):
    return np.einsum('i,j,k', c, b, a)


def calculate_Qi(Ui):
    val = []
    result = np.zeros((Ui.shape[1], Ui.shape[1]), dtype=float)
    for i in range(Ui.shape[1]):
        val.append(np.sum(Ui[:, i]))

    np.fill_diagonal(result, val)
    return result


def unfold(tensor, mode=0):
    """Returns the mode-`mode` unfolding of `tensor`
    """
    return np.transpose(tensor, reorder(range(tensor.ndim), mode)).reshape((tensor.shape[mode], -1))


def reorder(indices, mode):
    """Reorders the elements
    """
    indices = list(indices)
    element = indices.pop(mode)
    return ([element] + indices[::-1])


def kronecker_product(a, b):
    rowa = a.shape[0]
    cola = a.shape[1]
    rowb = b.shape[0]
    colb = b.shape[1]
    c = np.zeros((rowa * rowb, cola * colb), dtype=np.float16)

    # i loops till rowa
    for i in range(rowa):
        # k loops till rowb
        for k in range(rowb):
            # j loops till cola
            for j in range(cola):
                # l loops till colb
                for l in range(colb):
                    # Each element of matrix A is
                    # multiplied by whole Matrix B
                    # resp and stored as Matrix C
                    c[(i * rowb + k), (j * colb + l)] = a[i, j] * b[k, l]
                print('l finished')
            print('j finished ---')
        print('k finished --- ---')
    return c


if __name__ == '__main__':
    admm(10)
    a = np.array([[1, 3, 5],
                  [7, 9, 11]])
    b = np.array([[2, 4, 6],
                  [8, 10, 12]])
    print(kron(a, b))
    print(kronecker_product(a, b))