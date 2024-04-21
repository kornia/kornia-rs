# Compute Mean and Std of a dataset

This example demonstrates how to compute the mean and standard deviation of a dataset using `kornia-serve`.

## 1. Build `kornia-serve`

```bash
cd ~/kornia-serve
./build_service.sh  # Build the service in a Docker container
```

## 2. Run the service

```bash
./run_service.sh --data-dir /mnt/data
```

## 3. Install python dependencies

```bash
pip install uv  # Install the `uv` library
source .venv/bin/activate  # Activate the virtual environment
uv install -r requirements.txt  # Install the required python packages
```

## 3. Run the example

```bash
python client.py --images-dir /my/dataset --num-workers 16
```
