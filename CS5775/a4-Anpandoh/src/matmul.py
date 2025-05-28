import torch
import torch.nn.functional as F
import numpy as np


def matmul(A, B, method='naive', **kwargs):
    """
    Multiply two matrices.
    :param A: (N, M) torch tensor.
    :param B: (M, K) torch tensor.
    :param method:
    :return:
        Output matrix with shape (N, K)
    """
    method = method.lower()
    if method in ['naive', 'pytorch', 'torch']:
        return naive(A, B)
    elif method == 'svd':
        return svd(A, B, **kwargs)
    elif method in ['log', 'logmatmul']:
        return logmatmul(A, B, **kwargs)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(A, B, **kwargs):
    return A @ B


def svd(A, B, rank_A=None, rank_B=None):
    """
    Low‑rank SVD approximation for A and B before multiplying.
    The parameters match the original stub.
    """
    def _approx(mat, rank):
        if rank is None or rank >= min(mat.shape):
            return mat
        # Use PyTorch SVD when possible, otherwise fall back to NumPy
        try:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        except RuntimeError:
            U_np, S_np, Vh_np = np.linalg.svd(mat.cpu().numpy(), full_matrices=False)
            U, S, Vh = (torch.from_numpy(U_np).to(mat),
                        torch.from_numpy(S_np).to(mat),
                        torch.from_numpy(Vh_np).to(mat))
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        return (U_r * S_r) @ Vh_r   # same shape as original matrix

    A_hat = _approx(A, rank_A)
    B_hat = _approx(B, rank_B)
    return A_hat @ B_hat


def logmatmul(A, B, **kwargs):
    """
    Matrix multiplication using log-domain computation, supporting real values (including negatives).
    Args:
        A: (N, M) tensor
        B: (M, K) tensor
    Returns:
        Tensor of shape (N, K)
    """
    eps = kwargs.get("eps", 1e-20)

    # Decompose A and B into sign and log(abs)
    sign_A = torch.sign(A)
    sign_B = torch.sign(B)

    log_abs_A = torch.log(A.abs().clamp_min(eps))  # (N, M)
    log_abs_B = torch.log(B.abs().clamp_min(eps))  # (M, K)

    # Compute log-sum-exp of pairwise log-abs values
    # C_ij = sum_k A_ik * B_kj = sum_k sgn(A_ik)*sgn(B_kj)*exp(log|A_ik| + log|B_kj|)
    log_prod = log_abs_A.unsqueeze(2) + log_abs_B.unsqueeze(0)  # (N, M, K)

    # Sign product
    sign_prod = (sign_A.unsqueeze(2) * sign_B.unsqueeze(0))     # (N, M, K)

    # logsumexp over signed terms: handle positives and negatives separately
    pos_mask = sign_prod > 0
    neg_mask = sign_prod < 0

    # Compute log-sum-exp for positive and negative terms separately
    pos_logsumexp = torch.logsumexp(log_prod.masked_fill(~pos_mask, float('-inf')), dim=1)
    neg_logsumexp = torch.logsumexp(log_prod.masked_fill(~neg_mask, float('-inf')), dim=1)

    # Combine both using: result = pos_sum - neg_sum
    # This is stable unless pos ≈ neg
    pos_val = torch.exp(pos_logsumexp)
    neg_val = torch.exp(neg_logsumexp)
    return pos_val - neg_val