from layer_utils import *
from activations import GELU, Snake
from unet_2d import UNET_left, UNET_right
from backbone_zoo import backbone_zoo, bach_norm_checker


class UNET_att_right(nn.Module):
    def __init__(self,channel_in, channel_out, att_channel_in, att_channel_out, kernel_size=3, stack_num=2,
                   activation='ReLU', atten_activation='ReLU', attention='add',
                   unpool=True, batch_norm=False):
        super(UNET_att_right, self).__init__()
        pool_size = 2
        self.kernel_size = kernel_size
        self.stack_num = stack_num
        self.activation = activation
        self.batch_norm = batch_norm
        self.dec = DecodeLayer(channel_in, channel_out, pool_size, unpool, 
                     activation=activation, batch_norm=batch_norm)
        self.left = AttentionGate(channel_in=att_channel_in, channel_out=att_channel_out,activation=atten_activation, 
                            attention=attention)
        #self.cnv_stk = ConvStack(channel_out, kernel_size, stack_num=stack_num, activation=activation, 
             #      batch_norm=batch_norm)
        self.channel_out = channel_out
    def forward(self,x,x_left):
       # print(x.shape)
        x = self.dec(x)
        #print(x.shape)
        #print(x_left.shape)
        x_left = self.left(x_left,x)
       # print(x_left.shape)
        h = torch.cat([x,x_left], dim = 1)
        cnv_stk = ConvStack(h.shape[1],self.channel_out, self.kernel_size, stack_num=self.stack_num, activation=self.activation, 
                   batch_norm=self.batch_norm)
        h = cnv_stk(h)
        return h
    
def test_UNET_att_right():
    # Define parameters
    channel_in = 64
    channel_out = 128
    att_channel_in = 32
    att_channel_out = 32
    kernel_size = 3
    stack_num = 2
    activation = 'ReLU'
    atten_activation = 'ReLU'
    attention = 'add'
    unpool = True
    batch_norm = False

    # Create an instance of UNET_att_right
    model = UNET_att_right(channel_in, channel_out, att_channel_in,att_channel_out, kernel_size, stack_num,
                           activation, atten_activation, attention, unpool, batch_norm)

    # Create dummy inputs
    x = torch.randn(1, channel_in, 64, 64)       # (batch_size, channels, height, width)
    x_left = torch.randn(1, att_channel_in, 128, 128)  # (batch_size, channels, height, width)

    # Forward pass
    output = model(x, x_left)
    #print("Output: ", output.shape)
    # Check the output shape
    #assert output.shape == (1, channel_out, 64, 64), f"Expected shape (1, {channel_out}, 64, 64), but got {output.shape}"
#test_UNET_att_right()
class att_unet_2d_base(nn.Module):
    def __init__(self,input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                     activation='ReLU', atten_activation='ReLU', attention='add', batch_norm=False, pool=True, unpool=True, 
                     backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True):
        
        super(att_unet_2d_base,self).__init__()
        self.act_fn = getattr(nn, activation)()
        self.attention = attention
        self.atten_activation = atten_activation
        self.batch_norm = batch_norm
        self.activation = activation
        self.backbone = backbone
        self.input = input_tensor
        self.cnv_no_bck = ConvStack(input_tensor[0],filter_num[0], stack_num=stack_num_down, activation=activation, 
                       batch_norm=batch_norm)
        self.filter_num = filter_num
        self.unpool = unpool
        self.stack_num_up = stack_num_up
        self.stack_num_down = stack_num_down
        self.pool = pool
        self.weights = weights
        self.fb = freeze_backbone
        self.fbn = freeze_batch_norm
    def forward(self, x):
        depth_ = len(self.filter_num)
        X_skip = []
        if self.backbone is None:
            #x = self.input
            x = self.cnv_no_bck(x)
            X_skip.append(x)
            for i,f in enumerate(self.filter_num[1:]):
                model_unet_left = UNET_left(x.shape[1],f, stack_num=self.stack_num_down, activation=self.activation, pool=self.pool, 
                          batch_norm=self.batch_norm)
                x = model_unet_left(x)
                X_skip.append(x)
        else:
            #TODO: backbone thing
            if 'vgg' in self.backbone:
                backbone_ = backbone_zoo(self.backbone, self.weights, x.shape, depth_, self.fb, self.fbn)
                X_skip = backbone_([x,])
                depth_encode = len(X_skip)
            else:
                backbone_ = backbone_zoo(self.backbone, self.weights, x.shape, depth_-1, self.fb, self.fbn)
                X_skip = backbone_([x,])
                depth_encode = len(X_skip)+1
                
            if depth_encode< depth_:
                x = X_skip[-1]
                for i in range(depth_ - depth_encode):
                    i_real = i+depth_encode
                    model_left_unet = UNET_left(X.shape[1],self.filter_num[i_real], stack_num=self.stack_num_down, activation=self.activation, pool=self.pool, 
                              batch_norm=self.batch_norm)
                    x = model_left_unet(x)
                    X_skip.append(x)
        X_skip = X_skip[::-1]
        # upsampling begins at the deepest available tensor
        X = X_skip[0]
        # other tensors are preserved for concatenation
        X_decode = X_skip[1:]
        depth_decode = len(X_decode)

        # reverse indexing filter numbers
        filter_num_decode = self.filter_num[:-1][::-1]
        for i in range(depth_decode):
            f = filter_num_decode[i]
            model_unet_att_right = UNET_att_right(X.shape[1],f, f,att_channel_out=f//2, stack_num=self.stack_num_up,
                        activation=self.activation, atten_activation=self.atten_activation, attention=self.attention,
                        unpool=self.unpool, batch_norm=self.batch_norm)
            
            X = model_unet_att_right(X, X_decode[i])
            
        if depth_decode < depth_-1:
            for i in range(depth_-depth_decode-1):
                i_real = i + depth_decode
                model_right = UNET_right(X.shape[1],filter_num_decode[i_real], stack_num=self.stack_num_up, activation=self.activation, 
                    unpool=self.unpool, batch_norm=self.batch_norm, concat=False)
                X = model_right(X, None)
                
        return X
        
def test_base():
    input_tensor = torch.randn(1, 3, 256, 256)
    input_shape = (3, 256, 256)# Batch size of 1, 3 channels, 256x256 image
    filter_num = [64, 128, 256, 512]
    model = att_unet_2d_base(input_shape, filter_num)

    # Forward pass
    output = model(input_tensor)
    print("Output shape:", output.shape)
    
#test_base()
# it's returning a model I am returning an value need to check
class att_unet_2d(nn.Module):
    def __init__(self,input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2, activation='ReLU', 
                atten_activation='ReLU', attention='add', output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
                backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True):
        super(att_unet_2d, self).__init__()
        self.act_fn = getattr(nn, activation)
        if backbone is not None:
            bach_norm_checker(backbone, batch_norm)
            
        self.dsc = att_unet_2d_base(input_size, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                         activation=activation, atten_activation=atten_activation, attention=attention,
                         batch_norm=batch_norm, pool=pool, unpool=unpool, 
                         backbone=backbone, weights=weights, freeze_backbone=freeze_backbone, 
                         freeze_batch_norm=freeze_backbone)
        #self.conv_out = CONV_output(n_labels, kernel_size=1, activation=output_activation)
        self.n_labels = n_labels
        self.output_activation = output_activation
    def forward(self, x):
        x = self.dsc(x)
        conv_out = CONV_output(in_channels=x.shape[1], n_labels=self.n_labels, kernel_size=1, activation=self.output_activation)
        x = conv_out(x)
        return x
    
def test_att_unet_2d():
    input_size = (1,3, 256, 256)  # Batch size of 1, 3 channels, 256x256 image
    filter_num = [64, 128, 256, 512]
    n_labels = 10
    input_size1 = (3, 256, 256)
    input_tensor = torch.randn(input_size)
    # Instantiate the model
    model = att_unet_2d(input_size1, filter_num, n_labels)

    # Create a dummy input tensor
    input_tensor = torch.randn(input_size)

    # Perform a forward pass
    output = model(input_tensor)

    # Print the output shape
    print("Output shape:", output.shape)

# Run the test
"""test_att_unet_2d()"""