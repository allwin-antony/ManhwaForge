import paddle
print(f"PaddlePaddle Version: {paddle.__version__}")
print(f"Device: {paddle.device.get_device()}")
print(f"Is Compiled with CUDA: {paddle.is_compiled_with_cuda()}")
