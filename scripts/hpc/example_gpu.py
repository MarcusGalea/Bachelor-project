import torch

number_of_gpus = torch.cuda.device_count()
current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
output_string = f" Number of GPUs: {number_of_gpus}\n Name of GPU {current_gpu_name}"
f = open("output_gpu.txt", "a")
f.write(output_string)
f.close()
