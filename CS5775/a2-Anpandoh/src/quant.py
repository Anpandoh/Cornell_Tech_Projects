import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# Round function
class ste_round(torch.autograd.Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: fill-in (start)
        return grad_output.clone().detach()
        # TODO: fill-in (end)


# Get percnetile min max
def get_percentile_min_max(
        input, lower_percentile, upper_percentile, output_tensor=False):
    """
    Calculate the percentile max and min values in a given tensor
    Parameters:
    ----------
    input: tensor
        the tensor to calculate percentile max and min
    lower_percentile: float
        if 0.1, means we return the value of the smallest 0.1% value in the
        tensor as percentile min
    upper_percentile: float
        if 99.9, means we return the value of the largest 0.1% value in the
        tensor as percentile max
    output_tensor: bool, default False
        if True, this function returns tensors, otherwise it returns values
    """
    input_length = input.shape[0]

    lower_index = round(input_length * (1 - lower_percentile * 0.01))
    upper_index = round(input_length * upper_percentile * 0.01)

    upper_bound = torch.kthvalue(input, k=upper_index).values

    if lower_percentile == 0:
        lower_bound = upper_bound * 0
    else:
        lower_bound = -torch.kthvalue(-input, k=lower_index).values

    if not output_tensor:
        lower_bound = lower_bound.item()
        upper_bound = upper_bound.item()
    return lower_bound, upper_bound


def linear_quantize(input, scale, zero_point):
    """
    Quantize floating point input tensor to integers with the given scaling
    factor and zeropoint.
    Parameters:
    ----------
    input: floating point input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """
    # TODO: fill-in (start)
    output = input / scale + zero_point
    # output = torch.round(output)
    output = ste_round.apply(output)
    # TODO: fill-in (end)
    return output


def get_moving_avg_min_max(
        x, prev_x_min, prev_x_max,
        act_percentile=0., act_range_momentum=0.95, is_symmetric=False):
    if act_percentile == 0.:
        x_min = x.data.min()
        x_max = x.data.max()
    elif is_symmetric:
        x_min, x_max = get_percentile_min_max(
            x.detach().view(-1), 100 - act_percentile,
            act_percentile, output_tensor=True)
    # Note that our asymmetric quantization is implemented using scaled unsigned
    # integers without zero_points, that is to say our asymmetric quantization
    # should always be after ReLU, which makes the minimum value to be always 0.
    # As a result, if we use percentile mode for asymmetric quantization, the
    # lower_percentile will be set to 0 in order to make sure the final x_min is 0.
    elif not is_symmetric:
        x_min, x_max = get_percentile_min_max(
            x.detach().view(-1), 0, act_percentile, output_tensor=True)

    # Initialization
    if prev_x_min is None or prev_x_max is None:
        new_x_min = x_min
        new_x_max = x_max
    elif prev_x_min == prev_x_max:
        new_x_min = prev_x_min + x_min
        new_x_max = prev_x_max + x_max
    # use momentum to update the quantization range
    elif act_range_momentum == -1:
        new_x_min = min(prev_x_min, x_min)
        new_x_max = max(prev_x_max, x_max)
    else:
        new_x_min = prev_x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        new_x_max = prev_x_max * act_range_momentum + x_max * (1 - act_range_momentum)
    return new_x_min, new_x_max


# Function for symmetric quantization
class SymmetricQuantFunction(torch.autograd.Function):
    """
    Class to quantize the given floating-point values using quantization with
    given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None, specified_zero_point=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        specified_scale: pre-calculated scaling factor for the tensor x
        specified_zero_point: pre-calculated zero_point for the tensor x
        """
        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The QuantFunction requires a pre-calculated scaling factor")
        ctx.scale = scale


        zero_point = 0

        # TODO: fill-in (start)
        qmin = -(2 ** (k - 1))
        qmax = (2 ** (k - 1)) - 1
        output = ste_round.apply(x / scale).clamp(qmin, qmax)
        # Quantize the input using linear quantization
        
        # output = linear_quantize(x, scale, zero_point)
        
        # # Calculate the quantization range based on bitwidth
        # n = 2 ** (k - 1) - 1
        
        # # Clamp the output to ensure it's within the quantization range
        # output = torch.clamp(output, -n - 1, n)
        
        # # Scale back to the original range
        # output = (output - zero_point) * scale

        # TODO: fill-in (end)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        return grad_output.clone() / scale, None, None, None


# Function for asymmetric quantization
class AsymmetricQuantFunction(torch.autograd.Function):
    """
    Class to quantize the given floating-point values using quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None, specified_zero_point=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of QuantFunction requires pre-calculated scaling factor.
        The current hardware support requires quantization to use scaled unsigned integers
        without zero_point, so quantization is for activations after ReLU, and zero_point is set to 0.
        specified_scale: pre-calculated scaling factor for the tensor x
        specified_zero_point: pre-calculated zero_point for the tensor x
        """
        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The QuantFunction requires a pre-calculated scaling factor")
        ctx.scale = scale

        if specified_zero_point is not None:
            zero_point = specified_zero_point
        else:
            raise ValueError("The QuantFunction requires a pre-calculated zero point")

        # TODO: fill-in (start)
        qmin = 0
        qmax = (2 ** k) - 1
        output = ste_round.apply(x / scale + zero_point).clamp(qmin, qmax)
        # output = linear_quantize(x, scale, zero_point)
        # # Calculate the quantization range based on bitwidth
        # n = 2 ** k - 1
        
        # # Clamp the output to ensure it's within the quantization range
        # output = torch.clamp(output, 0, n)
        
        # # Scale back to the original range
        # output = (output - zero_point) * scale
        # TODO: fill-in (end)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        return grad_output.clone() / scale, None, None, None



class QConfig(object):
    """
    Type that contains configurations for quantization
    and stores calculated quantization parameters
    """

    def __init__(self, quant_bits=8, is_symmetric=True):
        self.quant_bits = quant_bits
        self.is_symmetric = is_symmetric
        self.quant_mode = 'symmetric' if self.is_symmetric else 'asymmetric'
        self.quantize_function = SymmetricQuantFunction.apply if self.is_symmetric \
            else AsymmetricQuantFunction.apply

        # States accumulators
        self.prev_scale = None
        self.prev_zeropoint = None
        self.prev_min = None
        self.prev_max = None

    def __repr__(self):
        s = f'quant_bits={self.quant_bits}, quant_mode={self.quant_mode}, ' + \
            f'prev_scale={self.prev_scale}, prev_zeropoint={self.prev_zeropoint}, ' + \
            f'prev_min={self.prev_min}, prev_max={self.prev_max} '
        return s

    def get_quantization_params(self, saturation_min, saturation_max):
        """
        Calculate scale and zero point given saturation_min and saturation_max
        """
        # with torch.no_grad():
        #     #TODO: fill-in (start):
        #     epsilon = 1e-8  # Small value to prevent division by zero
        #     if self.is_symmetric:
        #         # Find the max absolute value to determine the scale
        #         saturation_absmax = max(abs(saturation_min), abs(saturation_max))
        #         # Calculate scale with epsilon to prevent underflow
        #         scale = (saturation_absmax + epsilon) / ((2 ** (self.quant_bits - 1)) - 1)
        #         # Zero point is always 0 for symmetric quantization
        #         zero_point = torch.tensor(0, dtype=torch.float32)
        #     else:
        #         # Define quantization range
        #         qmin = 0
        #         qmax = (2 ** self.quant_bits) - 1
        #         # Calculate scale with epsilon to prevent underflow
        #         scale = (saturation_max - saturation_min + epsilon) / (qmax - qmin)
        #         # Calculate zero_point to shift the range
        #         zero_point = qmin - torch.round(saturation_min / scale)
        #         # Clamp zero_point to the valid range
        #         zero_point = torch.clamp(zero_point, qmin, qmax)
        #     #TODO: fill-in (end)

        epsilon = 1e-8
        with torch.no_grad():
            epsilon = 1e-8  # Small value to prevent division by zero
            if self.is_symmetric:
                # For symmetric quantization, we use the max absolute value
                max_abs = max(abs(saturation_min), abs(saturation_max))
                # Calculate the scale factor
                n = 2 ** (self.quant_bits - 1) - 1
                scale = torch.tensor((max_abs + epsilon) / n)
                # For symmetric quantization, zero point is 0
                zero_point = torch.tensor(0)
            else:
                # For asymmetric quantization
                n = 2 ** self.quant_bits - 1
                # Calculate scale factor
                scale = torch.tensor((saturation_max - saturation_min + epsilon) / n)
                # Calculate zero point
                zero_point = torch.tensor(-torch.round(saturation_min / scale).item())
        return scale, zero_point


    def quantize_with_params(self, x, scale, zero_point, fake_quantize=False):
        """
        Calculate quantized value given float value, scale, and zero point
        """
        x_q = self.quantize_function(x, self.quant_bits, scale, zero_point)
        if fake_quantize:
            x_q = (x_q - zero_point) * scale
        return x_q

    def quantize_with_min_max(self, x, saturation_min, saturation_max, fake_quantize=False):
        """
        Calculate quantized value given float value, saturation_min, and saturation_max
        """
        # Compute scale and zeropoint for quantization
        scale, zero_point = self.get_quantization_params(saturation_min, saturation_max)
        # Update and store min and max
        self.prev_min = saturation_min
        self.prev_max = saturation_max
        # Update and store computed scale and zero_point
        self.prev_scale = scale
        self.prev_zeropoint = zero_point

        x_q = self.quantize_with_params(x, scale, zero_point, fake_quantize=fake_quantize)
        return x_q

    def quantize_with_prev_params(self, x, fake_quantize=False):
        """
        Calculate quantized value using scale and zero point calculated in previous quantization
        """
        assert self.prev_scale is not None and \
               self.prev_zeropoint is not None, "no params saved"
        x_q = self.quantize_with_params(
            x, self.prev_scale, self.prev_zeropoint, fake_quantize=fake_quantize)
        return x_q

    def copy(self):
        """
        Create a new QConfig instance with the same quant_bits and is_symmetric setting
        """
        return QConfig(self.quant_bits, self.is_symmetric)


def quantize_activations(x, qconfig, is_moving_avg=False, fake_quantize=False):
    """
    Return the quantized activations (x) calculated using given qconfig.
    Moving average method is used to calculate min and max when 'is_moving_avg'
    is set to True. 'is_moving_avg' is usually set to True during training, and
    set to False during testing and validation
    """
    x_transform = x.data.detach()
    prev_x_min, prev_x_max = qconfig.prev_min, qconfig.prev_max

    if is_moving_avg:
        x_min, x_max = get_moving_avg_min_max(
            x_transform, prev_x_min, prev_x_max,
            act_percentile=99.9,
            is_symmetric=qconfig.is_symmetric)

    else:
        x_min, x_max = get_moving_avg_min_max(
            x_transform, None, None,
            act_percentile=99.9,
            is_symmetric=qconfig.is_symmetric)

    # Get quantized activations and update scale, zero_point, min, and max of qconfig
    x_q = qconfig.quantize_with_min_max(
        x, x_min, x_max, fake_quantize=fake_quantize)
    return x_q


def quantize_weights_bias(w, qconfig, fake_quantize=False):
    """
    Return quantized weights calculated using given qconfig.
    """

    # TODO: fill-in (start)
    # w_min = w.min()
    # w_max = w.max()
    
    # # Compute scale and zero point for quantization
    # scale, zero_point = qconfig.get_quantization_params(w_min, w_max)
    
    # # Update qconfig with the computed parameters
    # qconfig.prev_min = w_min
    # qconfig.prev_max = w_max
    # qconfig.prev_scale = scale
    # qconfig.prev_zeropoint = zero_point
    
    # # Quantize weights
    # w_q = qconfig.quantize_with_params(w, scale, zero_point, fake_quantize=fake_quantize)
    




    # Get the min and max values from the tensor
    w_min = w.data.min()
    w_max = w.data.max()
    
    # Calculate the quantization parameters (scale and zero point)
    scale, zero_point = qconfig.get_quantization_params(w_min, w_max)
    
    # Update qconfig parameters
    qconfig.prev_min = w_min
    qconfig.prev_max = w_max
    qconfig.prev_scale = scale
    qconfig.prev_zeropoint = zero_point
    
    # Quantize the weights
    w_q = qconfig.quantize_with_params(w, scale, zero_point, fake_quantize=fake_quantize)
    # TODO: fill-in (end)

    return w_q


def conv2d_linear_quantized(
        module, x, a_qconfig=None, w_qconfig=None, b_qconfig=None):
    """
    Calculate fake quantized output of conv2d or linear layer given the float
    module, input tensor, and quantization configurations for activation,
    weight, and bias.
    """
    assert isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d), \
        "the module cannot be quantized"
    assert a_qconfig is not None and w_qconfig is not None
    if module.bias is not None:
        assert b_qconfig is not None

    # TODO: fill-in (start)
    # Quantize input activations
    x_q = quantize_activations(x, a_qconfig, is_moving_avg=False, fake_quantize=True)
    
    # Quantize weights
    weight_q = quantize_weights_bias(module.weight, w_qconfig, fake_quantize=True)
    
    # Perform computation with quantized values
    if isinstance(module, nn.Conv2d):
        if module.bias is not None:
            # Quantize bias
            bias_q = quantize_weights_bias(module.bias, b_qconfig, fake_quantize=True)
            # Perform convolution with bias
            y = F.conv2d(
                x_q, 
                weight_q, 
                bias_q,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups
            )
        else:
            # Perform convolution without bias
            y = F.conv2d(
                x_q, 
                weight_q, 
                None,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups
            )
    else:  # nn.Linear
        if module.bias is not None:
            # Quantize bias
            bias_q = quantize_weights_bias(module.bias, b_qconfig, fake_quantize=True)
            # Perform linear operation with bias
            y = F.linear(x_q, weight_q, bias_q)
        else:
            # Perform linear operation without bias
            y = F.linear(x_q, weight_q)
 
    # TODO: fill-in (end)

    return y


class QuantWrapper(nn.Module):
    def __init__(self, module, a_qconfig, w_qconfig, b_qconfig):
        super(QuantWrapper, self).__init__()
        self.a_qconfig = a_qconfig
        self.w_qconfig = w_qconfig
        self.b_qconfig = b_qconfig
        self.module = module

    def __repr__(self):
        s = super().__repr__()[:-1]
        s = f'{s}'+ \
            f'\t(activation): {self.a_qconfig} \n' + \
            f'\t(weight): {self.w_qconfig} \n' + \
            f'\t(bias): {self.b_qconfig} \n)'
        return s

    def forward(self, x):
        out = conv2d_linear_quantized(
            self.module, x, self.a_qconfig, self.w_qconfig, self.b_qconfig)
        return out


def quantize_model(
        float_model, a_qconfig=None, w_qconfig=None, b_qconfig=None):
    """
    Making a float_model aware of quantization during training and inference
    """
    qat_model = copy.deepcopy(float_model)
    if a_qconfig is None or w_qconfig is None:
        return qat_model

    for name, layer in qat_model.named_children():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            quant_wrapper = QuantWrapper(
                layer, a_qconfig.copy(), w_qconfig.copy(), b_qconfig.copy())
            setattr(qat_model, f'{name}', quant_wrapper)

    return qat_model


def dequantize_model(qat_model):
    """
    Remove all QuantWrappers from qat_model without updating weights and
    activation to quantized values
    """
    float_model = copy.deepcopy(qat_model)
    for name, layer in float_model.named_children():
        if isinstance(layer, QuantWrapper):
            print(f'Layer to be deprepared: QuantWrapper: {name}')
            setattr(float_model, f'{name}', layer.module)
    return float_model
