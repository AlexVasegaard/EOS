def scipy_sparse_to_spmatrix(A):
    from cvxopt import spmatrix
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP
