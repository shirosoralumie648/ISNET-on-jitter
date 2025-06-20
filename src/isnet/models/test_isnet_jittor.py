import jittor as jt
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 使用绝对导入
from isnet.models.isnet_jittor import ISNet

# 禁用Jittor消息，如'generating source code'，使输出更干净
try:
    jt.flags.log_silent = 1  # 尝试使用当前版本的标志
    print('已设置Jittor日志为静音模式')
except Exception as e:
    print(f'警告: 无法设置Jittor日志标志: {e}')

def run_isnet_sanity_check():
    print("Running ISNet (Jittor) Sanity Check...")

    # Configuration parameters from ISNet.py's __main__ block
    layer_blocks = [4, 4, 4]
    channels = [8, 16, 32, 64] # stem_width, c1, c2, c3

    print(f"Using layer_blocks: {layer_blocks}")
    print(f"Using channels: {channels}")

    # Instantiate the ISNet model
    try:
        model = ISNet(layer_blocks=layer_blocks, channels=channels)
        print("ISNet model instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating ISNet model: {e}")
        return

    # Set model to evaluation mode (important for layers like BatchNorm, Dropout)
    model.eval()
    print("Model set to evaluation mode.")

    # Create dummy input tensors
    batch_size = 1
    img_height = 256
    img_width = 256
    
    # Dummy input image (e.g., NCHW format: Batch, Channels, Height, Width)
    dummy_x = jt.randn((batch_size, 3, img_height, img_width))
    # Dummy gradient map (e.g., from Sobel operator, also NCHW)
    dummy_x_grad = jt.randn((batch_size, 3, img_height, img_width))

    print(f"Dummy input x shape: {dummy_x.shape}")
    print(f"Dummy input x_grad shape: {dummy_x_grad.shape}")

    # Perform a forward pass
    try:
        print("Performing forward pass...")
        main_out, edge_out = model(dummy_x, dummy_x_grad)
        print("Forward pass completed.")

        # Print output shapes
        print(f"Main output shape: {main_out.shape}")
        print(f"Edge output shape: {edge_out.shape}")

        # Expected output shapes (for single channel output, same H, W as input)
        expected_shape = (batch_size, 1, img_height, img_width)
        if main_out.shape == expected_shape and edge_out.shape == expected_shape:
            print("Output shapes are as expected.")
        else:
            print(f"Warning: Output shapes mismatch! Expected {expected_shape}.")

    except Exception as e:
        print(f"Error during forward pass: {e}")

if __name__ == '__main__':
    run_isnet_sanity_check()
