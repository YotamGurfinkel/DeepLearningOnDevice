# PyTorch Networks

This folder contains a script to convert a PyTorch neural network to a TVM neural network that can run on an aarch64 processor

### To convert a network:

Edit the script to use the wanted PyTorch model then run:

```bash
python3 convert_pytorch.py name_of_output_network_file.tar
```
