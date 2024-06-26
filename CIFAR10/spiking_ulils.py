import numpy as np

def label_encoder(label, num_class):
    """
    function: on-hot encoding
    label: 0，1，2, ..., num_class-1
    num_class: number of class
    return: one-hot vector of m samples, shape=(m, num_class)
    """
    tmp = np.eye(num_class)
    return tmp[label]
    

class Conv2d():
    count = 0
    def __init__(self, in_channels, n_filter, filter_size, padding, stride, k=0, B=4, quant_level=0,\
                 use_ternary=False, use_int=False, first_layer=False):
        """
        function: spiking convolution with positive and negative dynamics
        parameters:
            in_channel: number of input channles
            n_filter: number of filter
            filter_size: filter size (h_filter, w_filter)
            padding: padding size
            stride: convolution stride
        """
        Conv2d.count += 1
        self.in_channels = in_channels
        self.n_filter = n_filter
        self.h_filter, self.w_filter = filter_size
        self.padding = padding
        self.stride = stride
        self.bound = np.zeros([n_filter])
        self.diff = np.zeros([n_filter])
        self.layer_num = Conv2d.count
        self.precisions = [k, B]
        self.quant_level = quant_level
        self.thresholds = np.zeros([n_filter])
        self.first_layer = first_layer
        self.debug = 1
        
        # initialize convolution parameters W, b
        self.W = np.random.randn(n_filter, self.in_channels, self.h_filter, self.w_filter) / np.sqrt(n_filter / 2.)
        self.b = np.zeros((n_filter, 1))
        
        self.params = [self.W, self.b, self.bound, self.diff, self.thresholds]
        self.type = 'conv'
        self.use_ternary = use_ternary
        self.use_int = use_int
        
    def __call__(self, X, time_step):
        # compute for output feature size
        self.n_x, _, self.h_x, self.w_x = X.shape
        self.h_out = (self.h_x + 2 * self.padding - self.h_filter) / self.stride + 1
        self.w_out = (self.w_x + 2 * self.padding - self.w_filter) / self.stride + 1
        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)

        # initialize neuron states at the first time step
        if time_step == 0:
            self.spike_collect = np.zeros([self.n_x, self.n_filter, self.h_out, self.w_out])
            self.spike_num = 0
            self.sop_num = 0
            self.mem = np.zeros([self.n_x, self.n_filter, self.h_out, self.w_out])
            self.thre_pos = np.repeat(np.repeat(np.repeat(self.bound[:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
            self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0) + np.repeat(np.repeat(np.repeat(self.diff[:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
            self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0)
            
            self.thre_neg = np.repeat(np.repeat(np.repeat(self.bound[:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
            self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0)
            self.thresholds = self.bound[:] + self.diff[:]

        self.spike_out = np.zeros([self.n_x, self.n_filter, self.h_out, self.w_out])

        
        # instance of Img2colIndices
        self.img2col_indices = Img2colIndices((self.h_filter, self.w_filter), self.padding, self.stride)
        
        return self.forward(X, time_step)
    
    def forward(self, X, time_step):
        # img2col for input feature
        self.x_col = self.img2col_indices.img2col(X)
        
        # reshape W
        self.w_row = self.W.reshape(self.n_filter, -1)
        
        # foward using matrix multiplication
        out = self.w_row @ self.x_col + self.b  
        out = out.reshape(self.n_filter, self.h_out, self.w_out, self.n_x)
        out = out.transpose(3, 0, 1, 2)
        
        # membrane potential accumulation
        if self.first_layer == True:
            self.mem = self.mem + out
        else:
            if self.use_int == True:
               self.mem = self.mem + out*10000
            else:
               self.mem = self.mem + out

        # compute difference
        self.mem_b = np.maximum(self.mem, np.repeat(np.repeat(np.repeat(self.bound[:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
            self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0))
        # generate spikes
        if self.first_layer == True:
           if time_step < (2**self.precisions[0])*self.precisions[1]:
               self.spike_out = np.where(self.mem_b >= np.repeat(np.repeat(np.repeat(self.thresholds[:][:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
                   self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0), 1, 0)*2**self.quant_level
               self.thresholds[:] = self.thresholds[:] + self.diff[:]
           else:
               self.spike_out = np.zeros([self.n_x, self.n_filter, self.h_out, self.w_out])

        else:
            #self.spike_out = np.where(self.mem_b > self.thre_pos, 1, np.where(self.mem_b < self.thre_neg, -1, 0))
            self.spike_out = np.where(self.mem_b > self.thre_pos, 1, np.where(self.mem_b < self.thre_neg + -1e-5, -1, 0))
            # update shadow membrane potential
            #注意扭转出错
            self.thre_pos_temp = self.thre_pos
            self.thre_neg_temp = self.thre_neg
            self.thre_pos = np.where(self.mem_b > self.thre_pos_temp, self.thre_pos + np.repeat(np.repeat(np.repeat(self.diff[:, np.newaxis], \
                self.w_out, axis=1)[:, np.newaxis, :], self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0), np.where(self.mem_b < self.thre_neg_temp + -1e-5, \
                self.thre_pos - np.repeat(np.repeat(np.repeat(self.diff[:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
                self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0), self.thre_pos))
        
            self.thre_neg = np.where(self.mem_b > self.thre_pos_temp, self.thre_neg + np.repeat(np.repeat(np.repeat(self.diff[:, np.newaxis], \
                self.w_out, axis=1)[:, np.newaxis, :], self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0), np.where(self.mem_b < self.thre_neg_temp + -1e-5, \
                self.thre_neg - np.repeat(np.repeat(np.repeat(self.diff[:, np.newaxis], self.w_out, axis=1)[:, np.newaxis, :], \
                self.h_out, axis=1)[np.newaxis, :, :, :], self.n_x, axis=0), self.thre_neg)) 

        # update spike collector
        self.spike_collect = self.spike_collect + self.spike_out
        # count spikes
        self.spike_num = self.spike_num + np.sum(np.abs(self.spike_out))
        # count synaptic operations for snn
        self.sop_num = self.sop_num + np.sum(np.abs(self.x_col)) * self.w_row.shape[0]
        #self.mem_trace[t] = self.mem_trace + np.abs(self.x_col)
        
        print('Time_step: ', time_step, 'Layer: ', self.layer_num, ', spike activities: ', np.sum(np.abs(self.spike_out)))

        self.debug = self.debug + 1
        
        return self.spike_out, self.spike_collect, self.spike_num, self.sop_num

    
class Img2colIndices():
    """
    a simple img2col implementation
    """
    def __init__(self, filter_size, padding, stride):
        """
        parameters:
            filter_shape: filter size (h_filter, w_filter)
            padding: padding size
            stride: convolution stride
        """
        self.h_filter, self.w_filter = filter_size
        self.padding = padding
        self.stride = stride
    
    def get_img2col_indices(self, h_out, w_out):
        """
        parameters:
            h_out: height of output feature
            w_out: width of output feature
        return:
            k: shape=(filter_height*filter_width*C, 1), index for channle
            i: shape=(filter_height*filter_width*C, out_height*out_width), index for row
            j: shape=(filter_height*filter_width*C, out_height*out_width), index for col
        """
        i0 = np.repeat(np.arange(self.h_filter), self.w_filter)
        i1 = np.repeat(np.arange(h_out), w_out) * self.stride
        i = i0.reshape(-1, 1) + i1
        i = np.tile(i, [self.c_x, 1])
        
        j0 = np.tile(np.arange(self.w_filter), self.h_filter)
        j1 = np.tile(np.arange(w_out), h_out) * self.stride
        j = j0.reshape(-1, 1) + j1
        j = np.tile(j, [self.c_x, 1])
        
        k = np.repeat(np.arange(self.c_x), self.h_filter * self.w_filter).reshape(-1, 1)
        
        return k, i, j
    
    def img2col(self, X):
        """
        parameters:
            x: input feature map，shape=(batch_size, channels, height, width)
        return:
            function of img2col, shape=(h_filter * w_filter*chanels, batch_size * h_out * w_out)
        """
        self.n_x, self.c_x, self.h_x, self.w_x = X.shape

        # compute for output feature size
        h_out = (self.h_x + 2 * self.padding - self.h_filter) / self.stride + 1
        w_out = (self.w_x + 2 * self.padding - self.w_filter) / self.stride + 1
        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception("Invalid dimention")
        else:
            h_out, w_out = int(h_out), int(w_out)
        
        # 0 padding
        x_padded = None
        if self.padding > 0:
            x_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = X
        
        # call img2col_indices
        self.img2col_indices = self.get_img2col_indices(h_out, w_out)
        k, i, j = self.img2col_indices
        
        # transpose
        cols = x_padded[:, k, i, j]  # shape=(batch_size, h_filter*w_filter*n_channel, h_out*w_out)
        cols = cols.transpose(1, 2, 0).reshape(self.h_filter * self.w_filter * self.c_x, -1)  # reshape
        
        return cols
    
    def col2img(self, cols):
        """
        col2img: inverse process for img2col
        parameters:
            cols: shape=(h_filter*w_filter*n_chanels, batch_size*h_out*w_out)
        """
        # reshape
        cols = cols.reshape(self.h_filter * self.w_filter * self.c_x, -1, self.n_x)
        cols = cols.transpose(2, 0, 1)
        
        h_padded, w_padded = self.h_x + 2 * self.padding, self.w_x + 2 * self.padding
        x_padded = np.zeros((self.n_x, self.c_x, h_padded, w_padded))
        
        k, i, j = self.img2col_indices
        
        np.add.at(x_padded, (slice(None), k, i, j), cols)
        
        if self.padding == 0:
            return x_padded
        else:
            return x_padded[:, :, self.padding : -self.padding, self.padding : -self.padding]


class BatchNorm2d():
    """
    batch normalization instance
    """
    count = 0
    def __init__(self, n_channel, momentum, use_int=False, k=0, B=4, first_layer=False):
        """
        parameters:
            n_channel: number of input feature channle
            momentum: moving_mean/moving_var momentum term
        """
        BatchNorm2d.count += 1
        self.layer_num = BatchNorm2d.count

        self.n_channel = n_channel
        self.momentum = momentum
        self.use_int = use_int
        self.first_layer = first_layer
        self.precisions = [k, B]
        
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = np.ones((1, n_channel, 1, 1))
        self.beta = np.zeros((1, n_channel, 1, 1))
        
        # initialization
        self.moving_mean = np.zeros((1, n_channel, 1, 1))
        self.moving_var = np.zeros((1, n_channel, 1, 1))
        
        self.params = [self.gamma, self.beta]
        self.debug = 10
        self.type = 'bn'
    
    def __call__(self, X, time_step, mode):
        """
        X: shape = (N, C, H, W)
        mode: train or test
        """
        self.X = X  
        return self.forward(X, time_step, mode)
    
    def forward(self, X, time_step, mode):
        """
        X: shape = (N, C, H, W)
        mode: mode: train or test
        """
        if mode != 'train':
            self.x_norm = (X - self.moving_mean) / np.sqrt(self.moving_var + 1e-5)
        else:
            # computing with multicast
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            self.var = X.var(axis=(0, 2, 3), keepdims=True)  
            self.x_norm = (X - mean) / (np.sqrt(self.var + 1e-5))
            
            # update moving_mean/moving_var
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.var
        # gamma and beta term
        out = self.x_norm * self.gamma + self.beta
        return out
    

class Flatten():
    """
    flatten function
    """
    count = 0
    def __init__(self):
        Flatten.count += 1
        self.layer_num = Flatten.count
        self.debug = 10
        self.type = 'flatten'

    def __call__(self, X, time_step):
        self.x_shape = X.shape # (batch_size, channels, height, width)
        
        return self.forward(X, time_step)
    
    def forward(self, X, time_step):
        out = X.transpose(0, 2, 3, 1)
        out = out.ravel().reshape(self.x_shape[0], -1)
        return out
    



class Linear():
    """
    linear layer or fully connected layer
    """
    count = 0
    def __init__(self, dim_in, dim_out, use_ternary=False):
        """
        parameters：
            dim_in: input size
            dim_out: output size
        """
        Linear.count += 1
        self.layer_num = Linear.count

        # initialization
        scale = np.sqrt(dim_in / 2)
        self.weight = np.random.standard_normal((dim_in, dim_out)) / scale
        self.bias = np.zeros(dim_out)
        # self.weight = np.random.randn(dim_in, dim_out)
        # self.bias = np.zeros(dim_out)
        
        self.params = [self.weight, self.bias]
        self.debug = 10
        self.type = 'fc'
        self.use_ternary = use_ternary
        
    def __call__(self, X, time_step):
        """
        parameters：
            X：input, shape=(batch_size, dim_in)
        return：
            xw + b
        """
        self.X = X
        return self.forward(time_step)
    
    def forward(self, time_step):
        return np.dot(self.X, self.weight) + self.bias
    

class Relu(object):
    """
    relu function
    """
    def __init__(self):
        self.X = None
    
    def __call__(self, X):
        self.X = X
        return self.forward(self.X)
    
    def forward(self, X):
        return np.maximum(0, X)
    

