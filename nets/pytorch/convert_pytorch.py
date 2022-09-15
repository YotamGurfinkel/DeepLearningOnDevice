import torchvision
import sys
import tvm
import torch


class SingleOutNet(torch.nn.Module):
    """A wrapper module to only return "out" from output dictionary in eval."""

    def __init__(self, network):
        """Initialize with the network that needs to be wrapped."""
        super().__init__()
        self.network = network

    def forward(self, x):
        """Only return the "out" of all the outputs."""
        return self.network.forward(x)["out"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"USAGE: python3 {sys.argv[0]} name_of_output_network_file.tar")
        exit(1)

    # REPLACE THS WITH YOUR PYTORCH NETWORK YOU WANT TO CONVERT
    network_to_convert = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        pretrained=True
    )

    # THE INPUT SIZE OF THE NETWORK
    input_size = 284

    model = SingleOutNet(network_to_convert).eval()

    input_shape = [1, 3, input_size, input_size]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    input_name = "input0"
    shape_list = [(input_name, tuple(input_data.shape))]

    mod, params = tvm.relay.frontend.from_pytorch(scripted_model, shape_list)
    target = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu")
    dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(mod, target=target, params=params)

    lib.export_library(sys.argv[1])
