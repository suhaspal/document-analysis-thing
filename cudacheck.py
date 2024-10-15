import torch

def check_and_set_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Set device to CUDA
        print("CUDA is available and set as the device!")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Sample tensor creation to verify device setting
    tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"Tensor is on device: {tensor.device}")

if __name__ == "__main__":
    check_and_set_cuda()
