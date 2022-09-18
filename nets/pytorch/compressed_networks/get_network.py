import torch
import torchvision
import torchprune as tp
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"USAGE: python3 {sys.argv[0]} ALDS_NET_CHECKPOINT_FILE")
        exit(1)

    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(num_classes=21)
    model.eval()
    saved = torch.load(sys.argv[1])

    model = tp.util.models.SingleOutNet(model)
    net = tp.util.net.NetHandle(model)
    x = torch.randn([1, 3, 513, 513])
    net.forward(x)
    pruned_net = tp.ALDSNet(net, None, None)
    pruned_net.load_state_dict(saved["net"], strict=False)
