# Solving MPCCs using CasADi and CCOpt
In order to provide users of languages other and Julia use of CCOpt, we developed an interface to the symbolic modelling and automatic differentiation tool: CasADi. 
This tutorial walks you through the process of installing the required tools, modelling your first MPCC in CasADi, and using CCOpt to solve it.
We will provide examples both for the Python and MATLAB interfaces for CasADi, and we assume that you have a working installation and a working knowledge of one of the two languages.

    !!! info
    This tutorial also assumes that the user is on an `x86-64` Linux platform. We additionally support `x86-64` Windows platforms and both `x86-64` and `arm64` based macos platforms.
    Additional information for these platforms can be found in [Appendix A: other platforms](@ref).

We advise python users to use your favorite virtual environment manager ([`virtualenv`](https://virtualenv.pypa.io/en/latest/), [`venv`](https://docs.python.org/3/library/venv.html), etc.) to set up your environment, as some environment variable manipulation is required.

## Installing the prerequisites:
### CasADi
In order to use the CCOpt interface in CasADi, you will need to install the [latest CasADi version](https://web.casadi.org/get/), and at least version 3.8.
For python users this can easily be done via PyPI:
```shell
pip install casadi>=3.8
```
whereas in matlab you will need to download the correct version and run:
```matlab
addpath('<yourpath>/casadi-3.8.2-windows64-matlab2018b')
```

In order to verify that installation was successful, in python run:
```python
from casadi import *
x = MX.sym("x")
print(jacobian(sin(x),x))
```
or in MATLAB:
```matlab
import casadi.*
x = MX.sym('x')
disp(jacobian(sin(x),x))
```
### libMad
The CasADi binaries ship with an plugin for CCOpt, but not the actual binaries. These are provided by `libMad.so` and it's bundled Julia runtime.
The library can be installed from [its GitHub releases page](https://github.com/madsuite-org/libMad/releases/latest).
Alternatively, you can install the latest release by dowloading and untar-ing the release:
```shell
curl -L https://github.com/madsuite-org/libMad/releases/download/<release-tag>/libMad-ubuntu-x86_64-<release-tag>.tar.gz --output libMad.tar.gz
tar -xzf libMad.tar.gz
```
in the location you wish to install the library to.
Finally, add the installation to the library path:
```shell
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:<path-to-libMad-install>/libMad-ubuntu-x86_64-v0.0.12-casadi/lib"
```

    !!! info
    Due to version conflicts between the host languages and Julia you may additionally need to preload specific versions of system libraries.
    #### Python
    ```shell
    export LD_PRELOAD="<path-to-libMad-install>/libMad-ubuntu-x86_64-v0.0.12-casadi/lib/julia/libcrypto.so"
    ```
    #### MATLAB
    ```shell
    export LD_PRELOAD="<path-to-libMad-install>/libMad-ubuntu-x86_64-v0.0.12-casadi/lib/julia/libunwind.so"
    ```
    
    More details can be found in the [README.md of libMad](https://github.com/madsuite-org/libMad).
    
### Modeling an MPCC
Now you are ready to model your first MPCC in CasADi! First we will model:
```math
  \begin{aligned}
    \min_{x\in\mathbb{R^3}} \; & (x_1-1)^2 + (x_2-2)^2 + (x_3-3)^2 \\
    \text{subject to} \quad &  x_1 - x_2 + 2*x_3 = 1\\
                            &  0\le x_1 \perp x_2 \ge 0\\
  \end{aligned}
```

which is a QPCC in vertical form.
This can be modelled in CasADi:
```python
import casadi as ca
import numpy as np

x = ca.SX.sym("x", 2)
x1 = x[0]
x2 = x[1]
x3 = x[2]

g = x1 - x2 + 2*x3

f = (x1-1)**2 + (x2-2)**2 + (x3-3)**2

nlp = {
    "x": x,
    "f": f,
    "g": g,
}
```
This generates all the symbolic variables, constrints, as well as the objective.

We then need to provide the CCOpt plugin with the information about the complementarity constraints:
```python
cc_pairs = np.array([[0,1]])
cc_types = np.array([0])

casadi_solver_opts = {
    "cc_pairs": cc_pairs.tolist(),
    "cc_types": cc_types.tolist(),
}
```
This is done above...

## Appendix A: other platforms