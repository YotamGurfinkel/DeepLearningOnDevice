# On Raspberry Pi

### This folder contains all code that runs on the Raspberry Pi itself, along with a script to convert the raw video that the Raspberry Pi produces

## Instructions

To run the code on a Raspberry Pi:

1. Compile `TVM runtime` using the `compile_tvm_runtime.sh` script

```sh
bash compile_tvm_runtime.sh
```

2. Add the following line to the `./tvm/python/tvm/__init__.py` file:

```python
from .contrib import graph_executor
```

This enables us to have the graph_executor api accessible with python, which is not accessible by default

3. Add this line to your `$HOME/.bashrc` file:

```sh
export PYTHONPATH=$PYTHONPATH:~/tvm/python
```

4. Install the required python libraries

```bash
pip install -r requirements.txt
```

5. Run the program on the Raspberry Pi, with a TELLO Drone wifi configured and specify a `.track` file, example file given

- The `.track` file syntax for each line is as given:
  `djitellopy_function_name_with_one_arg function_arg`

```sh
python3 main.py tracks/example_track.track
```
