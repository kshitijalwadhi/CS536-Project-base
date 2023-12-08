# CS536-Base

## First, create a virtual environment and install the necessary packages:

```
python3 -m venv .venv
source .venv_agbench/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Second, compile the grpc proto by running:

Follow this for installing grpc: `https://grpc.io/docs/languages/python/quickstart/`

`python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. object_detection.proto`

## Third, run the server as: 

`python3 server.py`

## Finally, run clients as:
`python3 client.py <fps>`
