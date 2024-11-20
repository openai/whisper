import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Define layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Example usage
if __name__ == "__main__":
    # Load Habana module for HPU support
    from habana_frameworks.torch.utils.library_loader import load_habana_module
    import habana_frameworks.torch.hpu as hthpu

    load_habana_module()

    device = None
    # Set device to HPU
    if hthpu.is_available():
        device = torch.device("hpu")
        print("Using HPU")

    if not device:
        print("HPU is not available")
        exit(1)

    # Create model instance and move it to the HPU
    model = SimpleCNN(num_classes=10).to(device)

    # Create a dummy input tensor and move it to the HPU
    input_tensor = torch.rand((64, 3, 224, 224), device=device)  # Batch size of 64

    # Forward pass through the model on HPU
    output = model(input_tensor)

    print("Output shape:", output.shape)  # Should be [64, num_classes]
