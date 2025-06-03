import jittor as jt
import jittor.nn as nn

class GetGradientNopadding(nn.Module):
    def __init__(self):
        super(GetGradientNopadding, self).__init__()
        kernel_v = jt.array([[0, -1, 0],
                             [0, 0, 0],
                             [0, 1, 0]], dtype='float32')
        kernel_h = jt.array([[0, 0, 0],
                             [-1, 0, 1],
                             [0, 0, 0]], dtype='float32')
        
        self.kernel_h = kernel_h.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, 3, 3)
        self.kernel_v = kernel_v.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, 3, 3)

    def execute(self, x):
        if x.shape[1] > 1:
            x_list = []
            for i in range(x.shape[1]):
                x_i = x[:, i:i+1, :, :] # Shape: (N, 1, H, W)
                grad_v = nn.conv2d(x_i, self.kernel_v, padding=1)
                grad_h = nn.conv2d(x_i, self.kernel_h, padding=1)
                grad_mag = jt.sqrt(grad_v * grad_v + grad_h * grad_h + 1e-6)
                x_list.append(grad_mag)
            output = jt.contrib.concat(x_list, dim=1)
        else: # 单通道输入
            grad_v = nn.conv2d(x, self.kernel_v, padding=1)
            grad_h = nn.conv2d(x, self.kernel_h, padding=1)
            output = jt.sqrt(grad_v * grad_v + grad_h * grad_h + 1e-6)
            
        return output
