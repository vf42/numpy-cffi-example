# Numpy + CFFI example

An example of using CFFI to wrap a C library for use with numpy, accompanied by a benchmark to evaluate the performance benefits.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: things will probably work fine with older versions of both numpy and cffi than the ones in requirements.txt, I just fixed the ones that happened to be installed for me.

## Running

Build the C extension:
```bash
python mylib_extension_build.py
```

Run tests:
```bash
python -m unittest myutils_test.py
```