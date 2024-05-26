from backbone_zoo import backbone_zoo, bach_norm_checker
from layer_utils import *

class UNET_left(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stack_num=2, activation='ReLU',
              pool=True, batch_norm=False):
        super(UNET_left, self).__init__()
        pool_size = 2
        self.enc = EncodeLayer(channel_in, channel_in, pool_size, pool, activation=activation, 
                     batch_norm=batch_norm)
        
        self.conv_st = ConvStack(channel_in, channel_out, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm)
        
    def forward(self, x):
        x = self.enc(x)
       # print("done")
        #print(x.shape)
        x = self.conv_st(x)
        return x
    
    
class discriminator_base(nn.Module):
    def __init__(self,input_size, filter_num, stack_num_down=2,
                       activation='ReLU', batch_norm=False, pool=True,
                       backbone=None):
        super(discriminator_base, self).__init__()
        self.conv = ConvStack(input_size[0],filter_num[0], stack_num=stack_num_down, activation=activation,
                   batch_norm=False)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.filter_num = filter_num
        self.stack_num_down = stack_num_down
        self.activation = activation
    def forward(self, x):
        X_skip = []
        x = self.conv(x)
        X_skip.append(x)
        for i,f in enumerate(self.filter_num[1:]):
            unet_left = UNET_left(x.shape[1],f, stack_num=self.stack_num_down, activation=self.activation, pool=False,
                        batch_norm=False)
            x = unet_left(x)
            
        print(x.shape)
        x = self.avg(x)
       # print("after avg: ",x.shape)
        ch = x.view(x.size(0), -1)
        print(ch.shape)
        x = DenseLayer(in_features=ch.shape[1], units=ch.shape[1], activation='LeakyReLU')(ch)  # First dense layer
        x = DenseLayer(in_features=ch.shape[1],units=1, activation=None)(x)  # LeakyReLU activation
        #x = self.dense_layer_2(x)  # Second dense layer
        x = torch.sigmoid(x)
        #print("here")
        return x
        
# this one is a little sus
class discriminator_2d(nn.Module):
    def __init__(self,input_size, filter_num, stack_num_down=2,
                     activation='ReLU', batch_norm=False, pool=False,
                     backbone=None):
        super(discriminator_2d, self).__init__()
        
        self.act_fn = getattr(nn, activation)
        if backbone is not None:
            bach_norm_checker(backbone, batch_norm)
            
        self.dsc = discriminator_base(input_size, filter_num, stack_num_down=stack_num_down,
                           activation=activation, batch_norm=batch_norm, pool=pool)
        
    def forward(self, x):
        x = self.dsc(x)
        return x
    
import torch
import torch.nn as nn

# Assuming the necessary classes and functions are imported or defined in your script.
# Since the imports are specific to your environment, they are assumed to be correct here.

def test_UNET_left():
    try:
        channel_in = 3
        channel_out = 64
        model = UNET_left(channel_in, channel_out, stack_num=2, activation='ReLU', pool=True, batch_norm=False)
        print(model)
        sample_input = torch.randn(1, 3, 64, 64)
        output = model(sample_input)
        print("UNET_left output shape:", output.shape)
        assert output.shape[1] == channel_out, "Output channel mismatch for UNET_left"
        print("UNET_left test passed!")
    except Exception as e:
        print(f"UNET_left test failed: {e}")

def test_discriminator_base():
    
    filter_num = [64, 128, 256, 512]
    sample_input = torch.randn(1, 3, 256, 256)
    input_size = (3, 256, 256)
    model = discriminator_base(input_size,filter_num, stack_num_down=2, activation='ReLU', batch_norm=False, pool=True)
    print(model)
    sample_input = torch.randn(1, 3, 256, 256)
    output = model(sample_input)
    print("discriminator_base output shape:", output.shape)
    assert output.shape == (sample_input.shape[0], 1), "Output shape mismatch for discriminator_base"
    print("discriminator_base test passed!")
    
def test_discriminator_2d():
    
        input_size = (3, 64, 64)
        filter_num = [64, 128, 256, 512]
        #model = discriminator_2d(input_size=input_size, filter_num=filter_num, stack_num_down=2,
                             #    activation='ReLU', batch_norm=False, pool=False, backbone=None)
        #print(model)
        sample_inputs = [torch.randn(1, 3, 64, 64), torch.randn(4, 3, 64, 64), torch.randn(8, 3, 64, 64)]
        for i, sample_input in enumerate(sample_inputs):
            model = discriminator_2d(input_size=input_size, filter_num=filter_num, stack_num_down=2,
                                 activation='ReLU', batch_norm=False, pool=False, backbone=None)
            output = model(sample_input)
            print(f"Test case {i+1}:")
            print("Input shape:", sample_input.shape)
            print("Output shape:", output.shape)
            print("Output:", output)
            assert output.shape == (sample_input.shape[0], 1), f"Output shape mismatch for test case {i+1}"
        print("discriminator_2d test passed!")
    

# Run all tests
"""test_UNET_left()
test_discriminator_base()
test_discriminator_2d()
"""