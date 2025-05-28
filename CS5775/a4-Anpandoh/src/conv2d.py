import torch
import torch.nn.functional as F


def conv2d(x, k, b, method='naive'):
    """
    Convolution of single instance and single input and output channel
    :param x:  (H, W) PyTorch Tensor
    :param k:  (Hk, Wk) PyTorch Tensor
    :param b:  (1,) PyTorch tensor or scalar
    :param method: Which method do we use to implement it. Valid choices include
                   'naive', 'torch', 'pytorch', 'im2col', 'winograd', and 'fft'
    :return:
        Output tensor should have shape (H_out, W_out)
    """
    method = method.lower()
    if method == 'naive':
        return naive(x, k, b)
    elif method in ['torch', 'pytorch']:
        return pytorch(x, k, b)
    elif method == 'im2col':
        return im2col(x, k, b)
    elif method == 'winograd':
        return winograd(x, k, b)
    elif method == 'fft':
        return fft(x, k, b)
    else:
        raise ValueError(f"Invalid [method] value: {method}")


def naive(x, k, b):
    """ Sliding window solution (O(HW Hk Wk)). """
    output_shape_0 = x.shape[0] - k.shape[0] + 1
    output_shape_1 = x.shape[1] - k.shape[1] + 1
    result = torch.zeros(output_shape_0, output_shape_1, dtype=x.dtype, device=x.device)
    for row in range(output_shape_0):
        for col in range(output_shape_1):
            window = x[row: row + k.shape[0], col: col + k.shape[1]]
            result[row, col] = torch.sum(window * k)
    return result + b


def pytorch(x, k, b):
    """ Thin wrapper around torch.nn.functional.conv2d (which is highly optimized C++/CUDA). """
    return F.conv2d(
        x.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        k.unsqueeze(0).unsqueeze(0),  # (1, 1, Hk, Wk)
        b  # (1,)
    ).squeeze(0).squeeze(0)  # (H_out, W_out)


def im2col(x, k, b):
    """Matrix multiply (GEMM) implementation using unfold (aka im2col).

    Complexity: O(H_out·W_out·Hk·Wk) but leverages BLAS for the inner product.
    """
    H, W = x.shape
    Hk, Wk = k.shape
    H_out = H - Hk + 1
    W_out = W - Wk + 1

    # Convert sliding windows to columns
    cols = F.unfold(x.unsqueeze(0).unsqueeze(0), kernel_size=(Hk, Wk))  # (1, Hk*Wk, H_out*W_out)
    cols = cols.squeeze(0).T  # (H_out*W_out, Hk*Wk)

    # Flatten kernel and multiply (broadcast bias later)
    kernel_flat = k.flatten()  # (Hk*Wk,)
    out = cols @ kernel_flat  # (H_out*W_out,)
    out = out.view(H_out, W_out)
    return out + b


def winograd(x, k, b):
    if k.shape != (3, 3):
        # Fallback to im2col if kernel is not 3x3
        return im2col(x, k, b)

    # Winograd transform matrices (cookbook form)
    G = x.new_tensor([
        [1, 0, 0],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0, 0, 1],
    ])  # (4,3)
    B = x.new_tensor([
        [1, 0, -1, 0],
        [0, 1, 1, 0],
        [0, -1, 1, 0],
        [0, 1, 0, -1],
    ])  # (4,4)
    A = x.new_tensor([
        [1, 1, 1, 0],
        [0, 1, -1, -1],
    ])  # (2,4)

    # Pre‑transform kernel (U = G * g * G^T)
    U = G @ k @ G.t()  # (4,4)

    H, W = x.shape
    H_out = H - 2  # because kernel is 3x3 (valid conv)
    W_out = W - 2
    n_tiles_h = (H_out + 1) // 2  # ceil division
    n_tiles_w = (W_out + 1) // 2

    # Pad input so every tile is complete (4x4 each)
    pad_h = n_tiles_h * 2 + 2 - H  # needed rows on bottom
    pad_w = n_tiles_w * 2 + 2 - W  # needed cols on right
    x_pad = F.pad(x, (0, pad_w, 0, pad_h))  # pad (left,right,top,bottom) — here only right & bottom

    output = torch.empty(H_out, W_out, dtype=x.dtype, device=x.device)

    for ti in range(n_tiles_h):
        for tj in range(n_tiles_w):
            d = x_pad[ti * 2 : ti * 2 + 4, tj * 2 : tj * 2 + 4]  # 4x4 tile
            V = B @ d @ B.t()        # transform input tile (4,4)
            M = U * V                # elementwise multiply in Winograd domain
            Y = A @ M @ A.t()
            
            # Write the valid part to the output tensor (crop if on boundary)
            row = ti * 2
            col = tj * 2
            h_lim = min(2, H_out - row)
            w_lim = min(2, W_out - col)
            output[row : row + h_lim, col : col + w_lim] = Y[:h_lim, :w_lim]

    return output + b


def fft(x, k, b):
    H, W = x.shape
    KH, KW = k.shape

    # Pad size to match full convolution
    H_pad = H + KH - 1
    W_pad = W + KW - 1

    # Pad input and flipped kernel to same size
    x_padded = torch.zeros((H_pad, W_pad), dtype=x.dtype, device=x.device)
    x_padded[:H, :W] = x

    k_flipped = torch.flip(k, dims=[0, 1])
    k_padded = torch.zeros((H_pad, W_pad), dtype=k.dtype, device=k.device)
    k_padded[:KH, :KW] = k_flipped

    # Perform FFTs
    X_fft = torch.fft.rfft2(x_padded)
    K_fft = torch.fft.rfft2(k_padded)

    # Element-wise multiply in frequency domain
    Y_fft = X_fft * K_fft

    # Inverse FFT
    y_full = torch.fft.irfft2(Y_fft, s=(H_pad, W_pad))

    # Extract valid region (same as torch's conv2d valid output)
    OH = H - KH + 1
    OW = W - KW + 1
    out = y_full[KH - 1 : KH - 1 + OH, KW - 1 : KW - 1 + OW]

    return out + b