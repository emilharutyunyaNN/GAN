from activations import GELU, Snake

import torch
import torch.nn as nn
import torch.functional as F

import torch
import torch.nn as nn

class DecodeLayer(nn.Module):
    def __init__(self, channel_in, channel_out, pool_size, unpool, kernel_size=3, activation='ReLU', batch_norm=False, name='decode'):
        super(DecodeLayer, self).__init__()
        
        if unpool is False:
            bias_flag = not batch_norm
        elif unpool == 'nearest':
            unpool = True
            interp = 'nearest'
        elif (unpool is True) or (unpool == 'bilinear'):
            unpool = True
            interp = 'bilinear'
        else:
            raise ValueError('Invalid unpool keyword')
        
        self.unpool = unpool
        self.batch_norm = batch_norm
        self.activation = activation
        self.pool_size = pool_size

        if self.unpool:
            self.up = nn.Upsample(scale_factor=pool_size, mode=interp)
        else:
            if kernel_size == 'auto':
                kernel_size = pool_size
            self.trans_conv = nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, stride=(pool_size, pool_size), padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2), output_padding=(pool_size-1))
            
            if self.batch_norm:
                self.bn = nn.BatchNorm2d(channel_out)
            if self.activation is not None:
                self.act_fn = getattr(nn, activation)()
                
    def forward(self, x):
        if self.unpool:
            x = self.up(x)
        else:
            x = self.trans_conv(x)
            if self.batch_norm:
                x = self.bn(x)
            if self.activation is not None:
                x = self.act_fn(x)
                
        return x

# Example usage
#only works for when pool_size is 2
"""pool_size = 2
channel = 64
name = 'decode'
activation = 'ReLU'
batch_norm = True

decode_layer = DecodeLayer(channel=channel, pool_size=pool_size, unpool=False, activation=activation, batch_norm=batch_norm)

# Create a dummy input tensor
input_tensor = torch.randn(1, channel, 16, 16)  # Batch size of 1, 64 channels, 16x16 spatial dimensions
output_tensor = decode_layer(input_tensor)

print(output_tensor.shape)"""  # Should print torch.Size([1, 64, 32, 32]) if using ConvTranspose2d


import torch
import torch.nn as nn

class EncodeLayer(nn.Module):
    def __init__(self, channel_in, channel_out, pool_size, pool, kernel_size='auto', 
                 activation='ReLU', batch_norm=False):
        super(EncodeLayer, self).__init__()
        
        if pool not in [False, True, 'max', 'ave']:
            raise ValueError('Invalid pool keyword')
            
        # maxpooling2d as default
        if pool is True:
            pool = 'max'
            
        elif pool is False:
            # stride conv configurations
            bias_flag = not batch_norm
        
        self.pool = pool
        self.batch_norm = batch_norm
        self.activation = activation
        
        if pool == 'max':
            self.pl = nn.MaxPool2d(kernel_size=(pool_size, pool_size))
            self.conv = None
            self.bn = None
            self.act_fn = None
        elif pool == 'ave':
            self.pl = nn.AvgPool2d(kernel_size=(pool_size, pool_size))
            self.conv = None
            self.bn = None
            self.act_fn = None
        else:
            self.pl = None
            if kernel_size == 'auto':
                kernel_size = pool_size
                
            self.conv = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=(pool_size, pool_size), padding=0)
            if batch_norm:
                self.bn = nn.BatchNorm2d(channel_out)
            else:
                self.bn = None
            if activation is not None:
                self.act_fn = getattr(nn, activation)()
            else:
                self.act_fn = None

    def forward(self, x):
        if self.pool in ['max', 'ave']:
            x = self.pl(x)
        else:
            x = self.conv(x)
            if self.bn:
                x = self.bn(x)
            if self.act_fn:
                x = self.act_fn(x)
                
        return x
    
# Test cases for the EncodeLayer class
def test_encode_layer():
    # Parameters
    channel = 64
    pool_size = 2
    input_tensor = torch.randn(1, channel, 32, 32)  # Batch size of 1, 64 channels, 32x32 spatial dimensions
    
    # Case 1: Max pooling
    encode_layer_max = EncodeLayer(channel=channel, pool_size=pool_size, pool='max')
    output_max = encode_layer_max(input_tensor)
    print(f"Max Pooling Output Shape: {output_max.shape}")  # Expected shape: [1, 64, 16, 16]

    # Case 2: Average pooling
    encode_layer_ave = EncodeLayer(channel=channel, pool_size=pool_size, pool='ave')
    output_ave = encode_layer_ave(input_tensor)
    print(f"Average Pooling Output Shape: {output_ave.shape}")  # Expected shape: [1, 64, 16, 16]

    # Case 3: Strided convolution with batch norm and activation
    encode_layer_conv_bn_act = EncodeLayer(channel=channel, pool_size=pool_size, pool=False, batch_norm=True, activation='ReLU')
    output_conv_bn_act = encode_layer_conv_bn_act(input_tensor)
    print(f"Strided Conv + BN + Act Output Shape: {output_conv_bn_act.shape}")  # Expected shape: [1, 64, 16, 16]

    # Case 4: Strided convolution without batch norm and activation
    encode_layer_conv = EncodeLayer(channel=channel, pool_size=pool_size, pool=False, batch_norm=False, activation=None)
    output_conv = encode_layer_conv(input_tensor)
    print(f"Strided Conv Output Shape: {output_conv.shape}")  # Expected shape: [1, 64, 16, 16]

# Run test cases
#test_encode_layer()


class AttentionGate(nn.Module):
    def __init__(self, channel_in,channel_out,  
                   activation='ReLU', 
                   attention='add'):
        super(AttentionGate, self).__init__()
        self.act_fn = getattr(nn, activation)()
        self.attention = getattr(torch, attention)
        self.theta_att = nn.Conv2d(channel_in, channel_out, kernel_size=1, bias=True)
        self.phi_g = nn.Conv2d(channel_in, channel_out, kernel_size=1, bias=True)
        self.psi_f = nn.Conv2d(channel_out, 1, kernel_size=1, bias=True)
        
    def forward(self,x,g):
        theta = self.theta_att(x)
        phi_g = self.phi_g(x)
        query = self.attention(theta, phi_g)
        f = self.act_fn(query)
        psi_f = self.psi_f(f)
        coef_att = torch.sigmoid(psi_f)
        x_att = x*coef_att
        #print("att: ", x_att.shape)
        return x_att
    
def test_attention_gate():
    # Parameters
    channel = 64
    height = 32
    width = 32

    # Create random input tensors
    X = torch.randn(1, channel, height, width)
    g = torch.randn(1, channel, height, width)

    # Instantiate the AttentionGate module
    attention_gate = AttentionGate(channel=channel, activation='ReLU', attention='add')

    # Forward pass
    output = attention_gate(X, g)

    # Print output shape
    print(f"Output shape: {output.shape}")

# Run test
#test_attention_gate()

import torch
import torch.nn as nn

class ConvStack(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stack_num=2, dilation_rate=1, activation='ReLU', batch_norm=False):
        super(ConvStack, self).__init__()
        self.bias_flag = not batch_norm
        self.batch_norm = batch_norm
        self.activation = getattr(nn, activation)()
        self.stack_num = stack_num
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.dr = dilation_rate

        # Define convolutional layers
        self.conv_layers = self._make_layers()

    def _make_layers(self):
        layers = []
        print(self.channel_in, self.channel_out)
        layers.append(nn.Conv2d(self.channel_in, self.channel_out, kernel_size=self.kernel_size, padding=1, bias=self.bias_flag, dilation=self.dr))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(self.channel_out))
        layers.append(self.activation)
        for _ in range(1,self.stack_num):
            layers.append(nn.Conv2d(self.channel_out, self.channel_out, kernel_size=self.kernel_size, padding=1, bias=self.bias_flag, dilation=self.dr))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(self.channel_out))
            layers.append(self.activation)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_layers(x)

# Define test cases
def test_basic():
    model = ConvStack(channel_in=32, channel_out=32)
    input_tensor = torch.randn(1, 32, 64, 64)  # Batch size 1, 32 channels, 64x64 image
    output = model(input_tensor)
    print("Test Case 1 Output Shape:", output.shape)

def test_with_batch_norm():
    model = ConvStack(channel_in=16, channel_out = 16, batch_norm=True)
    input_tensor = torch.randn(1, 16, 64, 64)  # Batch size 1, 16 channels, 64x64 image
    output = model(input_tensor)
    print("Test Case 2 Output Shape:", output.shape)

def test_custom_channels_and_stack_size():
    model = ConvStack(channel=16, stack_num=3)
    input_tensor = torch.randn(1, 16, 64, 64)  # Batch size 1, 16 channels, 64x64 image
    output = model(input_tensor)
    print("Test Case 3 Output Shape:", output.shape)

# Run test cases
#test_basic()
#test_with_batch_norm()
#test_custom_channels_and_stack_size()


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Res_CONV_stack(nn.Module):
    def __init__(self, channel_in, channel_out, res_num, activation='ReLU', batch_norm=False):
        super(Res_CONV_stack, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.res_num = res_num
        self.activation = getattr(nn, activation)()
        self.batch_norm = batch_norm

        # Define convolutional layers
        self.conv_stack = self._make_conv_stack()

    def _make_conv_stack(self):
        return ConvStack(channel_in=self.channel_in, channel_out=self.channel_out, stack_num=self.res_num, activation='Identity', batch_norm=self.batch_norm)

    def forward(self, x, x_skip):
        residual = self.conv_stack(x)
        output = x_skip + residual
        output = self.activation(output)
        return output


# Test the Res_CONV_stack class
def test_Res_CONV_stack_with_ConvStack():
    # Create input tensors
    input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
    skip_tensor = torch.randn(1, 3, 64, 64)   # Identity path tensor with the same shape

    # Instantiate Res_CONV_stack
    res_conv_stack = Res_CONV_stack(channel_in=3, channel_out=3, res_num=2, activation='ReLU', batch_norm=True)

    # Pass input through Res_CONV_stack
    output_tensor = res_conv_stack(input_tensor, skip_tensor)

    # Check output shape
    print("Output Shape:", output_tensor.shape)

# Run the test
#test_Res_CONV_stack_with_ConvStack()

#TODO: conv3d_to_2d to be done
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3D_z_valid(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, z_kernel_size, padding, dilation_rate, name):
        super(Conv3D_z_valid, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, z_kernel_size),
                                padding=padding, dilation=dilation_rate)
        self.zks = z_kernel_size
    def forward(self, x):
        x = self.conv3d(x)
        z_pad = self.zks // 2
        x = x[:, :, :, z_pad:-z_pad, :]
        return x

class CONV_stack_3D_to_2D(nn.Module):
    def __init__(self, channel, kernel_size=3, z_kernel_size=3, stack_num=2, dilation_rate=1, activation='ReLU',
                 batch_norm=False, name='conv_stack'):
        super(CONV_stack_3D_to_2D, self).__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        self.z_kernel_size = z_kernel_size
        self.stack_num = stack_num
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.batch_norm = batch_norm
        self.name = name
        self.conv_layers = self._make_layers()

    def _make_layers(self):
        layers = []
        bias_flag = not self.batch_norm
        for i in range(self.stack_num):
            layers.append(Conv3D_z_valid(self.channel, self.channel, self.kernel_size, self.z_kernel_size,
                                         padding='same', dilation_rate=self.dilation_rate,
                                         name='{}_{}'.format(self.name, i)))
            if self.batch_norm:
                layers.append(nn.BatchNorm3d(self.channel))
            activation_func = getattr(nn, self.activation)()
            layers.append(activation_func)
        return nn.Sequential(*layers)

    def forward(self, x):
        X_input = x
        x = torch.unsqueeze(x, dim=-1)  # Expand the dimensions to include the depth channel
        x = self.conv_layers(x)
        # Residual connection
        depth = X_input.size(-1)
        tmp = F.pad(X_input, (0, 0, 0, 0, 0, self.channel - 1), mode='constant', value=0)
        x = x + tmp[:, :, :, depth // 2:depth // 2 + 1, :]
        x = torch.squeeze(x, dim=-1)  # Remove the last dimension
        return x

# Test the CONV_stack_3D_to_2D
def test_CONV_stack_3D_to_2D():
    # Create an input tensor with shape [batch_size, channels, depth, height, width]
    input_tensor = torch.randn(1, 3, 32, 64, 64)  # Batch size 1, 3 channels, depth 32, height 64, width 64

    # Instantiate CONV_stack_3D_to_2D
    conv_stack = CONV_stack_3D_to_2D(channel=3, kernel_size=3, z_kernel_size=3, stack_num=2, dilation_rate=1,
                                      activation='ReLU', batch_norm=True)

    # Pass input through CONV_stack_3D_to_2D
    output_tensor = conv_stack(input_tensor)

    # Print the shape of the output tensor
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)

# Run the test
#test_CONV_stack_3D_to_2D()


class DepthwiseConv2d(torch.nn.Conv2d):
    def __init__(self,
                 in_channels,
                 depth_multiplier=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros'
                 ):
        out_channels = in_channels * depth_multiplier
        padding = (dilation * (kernel_size - 1)) // 2
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )
        
class Sep_CONV_stack(nn.Module):
    def __init__(self, channel, kernel_size=3, stack_num=1, dilation_rate=1, activation='ReLU', batch_norm=False):
        super(Sep_CONV_stack, self).__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        self.stack_num = stack_num
        self.dr = dilation_rate
        self.act_fn = getattr(nn, activation)()
        self.batch_norm = batch_norm
        self.conv_layers = self._make_layers()

    def _make_layers(self):
        layers = []
        for _ in range(self.stack_num):
            layers.append(DepthwiseConv2d(self.channel, kernel_size=self.kernel_size, padding=1, bias=not self.batch_norm, dilation=self.dr))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(self.channel))  # Corrected here
            layers.append(self.act_fn)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_layers(x)
        
def test_Sep_CONV_stack():
    # Create an input tensor with shape [batch_size, channels, height, width]
    input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, height 64, width 64

    # Instantiate Sep_CONV_stack
    sep_conv_stack = Sep_CONV_stack(channel=3, kernel_size=3, stack_num=2, dilation_rate=6, activation='ReLU', batch_norm=True)

    # Pass input through Sep_CONV_stack
    output_tensor = sep_conv_stack(input_tensor)

    # Print the shape of the output tensor
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)

# Run the test
#test_Sep_CONV_stack()

class ASPP_conv(nn.Module):
    def __init__(self, channel, activation='ReLU', batch_norm=True):
        super(ASPP_conv, self).__init__()
        self.act_fn = getattr(nn, activation)()
        self.bias_flag = not batch_norm
        self.channel = channel
        self.batch_norm = batch_norm
        
    def forward(self, x):
        shape_before = x.shape
        print(x.shape)
        b4 = nn.AdaptiveAvgPool2d(1)(x)
        #b4 = b4.unsqueeze(1).unsqueeze(1)
        print(b4.shape)
        b4 = nn.Conv2d(self.channel, 1, kernel_size=1, padding=1)(b4)
        if self.batch_norm:
            b4 = nn.BatchNorm2d(1)(b4)
        b4 = self.act_fn(b4)
        b4 = F.interpolate(b4, size=shape_before[2:], mode='bilinear', align_corners=True)
        
        b0 = nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=not self.batch_norm)(x)
        if self.batch_norm:
            b0 = nn.BatchNorm2d(self.channel)(b0)
        b0 = self.act_fn(b0)
        
        b_r6_mod = Sep_CONV_stack(self.channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=6, batch_norm=True)
        b_r6 = b_r6_mod(x)
        b_r9_mod = Sep_CONV_stack(self.channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=9, batch_norm=True)
        b_r9 = b_r9_mod(x)
        b_r12_mod = Sep_CONV_stack(self.channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=12, batch_norm=True)
        b_r12 = b_r12_mod(x)
        print(b0.shape, b4.shape, b_r6.shape, b_r9.shape, b_r12.shape)
        return torch.cat([b4, b0, b_r6, b_r9, b_r12], dim = 1)

    
def test_ASPP_conv():
    # Create an input tensor with shape [batch_size, channels, height, width]
    input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, height 64, width 64

    # Instantiate ASPP_conv module
    aspp_conv = ASPP_conv(channel=3, activation='ReLU', batch_norm=True)

    # Pass input through ASPP_conv
    output_tensor = aspp_conv(input_tensor)

    # Print the shape of the output tensor
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)

# Run the test
#test_ASPP_conv()

import torch
import torch.nn as nn

class CONV_output(nn.Module):
    def __init__(self, in_channels, n_labels, kernel_size=1, activation='Softmax'):
        super(CONV_output, self).__init__()
        self.conv = nn.Conv2d(in_channels, n_labels, kernel_size, padding=kernel_size // 2, bias=True)
        print("In:", in_channels)
        if activation:
            if activation == 'Sigmoid':
                self.act_fn = nn.Sigmoid()
            elif activation == 'Softmax':
                self.act_fn = nn.Softmax(dim=1)  # Softmax is usually applied along the channel dimension
            else:
                self.act_fn = getattr(nn, activation)()
        else:
            self.act_fn = None

    def forward(self, x):
        x = self.conv(x)
        if self.act_fn:
            x = self.act_fn(x)
        return x

def test_CONV_output():
    # Create an input tensor with shape [batch_size, channels, height, width]
    input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, height 64, width 64

    # Instantiate CONV_output
    conv_output = CONV_output(in_channels=3, n_labels=10, kernel_size=1, activation='Softmax')

    # Pass input through CONV_output
    output_tensor = conv_output(input_tensor)

    # Print the shape of the output tensor
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)

# Run the test
#test_CONV_output()

import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_features, units, activation='LeakyReLU'):
        super(DenseLayer, self).__init__()
        print(in_features, units)
        self.dense = nn.Linear(in_features, units)
        self.activation = getattr(nn, activation)() if activation else None
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.dense.weight)  # Xavier/Glorot initialization
        nn.init.zeros_(self.dense.bias)  # Zero initialization for bias

    def forward(self, x):
        print(x.shape)
        x = self.dense(x)
        print("dense :", x.shape)
        if self.activation:
            x = self.activation(x)
        return x

# Test the DenseLayer
def test_DenseLayer():
    input_tensor = torch.randn(1, 10)  # Batch size 1, input features 10
    dense_layer = DenseLayer(in_features=10, units=20, activation='LeakyReLU')
    output_tensor = dense_layer(input_tensor)
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)

# Run the test
#test_DenseLayer()
