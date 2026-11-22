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
    
## Modeling an MPCC
Now you are ready to model your first MPCC in CasADi! 

First we will model:
```math
  \begin{aligned}
    \min_{x\in\mathbb{R^3}} \; & (x_1-1)^2 + (x_2-2)^2 + (x_3-3)^2 \\
    \text{subject to} \quad &  x_1 - x_2 + 2*x_3 = 1\\
                            &  0\le x_1 \perp x_2 \ge 0\\
  \end{aligned}
```

which is a QPCC in vertical form, in python.
This can be modelled using CasADi in the following way:
```python
import casadi as ca
import numpy as np

x = ca.SX.sym("x", 3)
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

x0=[2.0, 2.0, 2.0],
lbx=[0.,0.,-np.inf],
ubx=[np.inf, np.inf, np.inf],
lbg=[1],
ubg=[1]
```
This generates all the symbolic variables, constrints, as well as the objective. Additionally we create the necessary bounds.

We then need to provide the CCOpt plugin with the information about the complementarity constraints.
This is done by passing a list of pairs of indices, along with a list of complementarity types: `VARVAR=0`, `VARCON=1`, `CONVAR=2` `CONCON=3`.
These correspond to complmentarity between two variables, a variable and an expression, or two expressions, and as such define which vector `x` or `g` the indices correspond to:
```python
cc_pairs = np.array([[0,1]])
cc_types = np.array([0])

casadi_solver_opts = {
    "cc_pairs": cc_pairs.tolist(),
    "cc_types": cc_types.tolist(),
}
```
In this case we have a complemntarity constraints between `x[0]` and `x[1]`.

    !!! Note
    The vector `cc_pairs` is a ``n_{cc}\times 2`` vector and is zero indexed, in python _and_ in MATLAB!

We then can pass options to the CCOpt solver, in the standard CasADi way. That is to say as dictionary entries in the options dictionary:
```python
casadi_solver_opts["ccopt"] = {
    "relaxation_update.TYPE": "RolloffRelaxationUpdate",
    "q_regularization": "critical_rho",
}
casadi_solver_opts["madnlp"] = {
    "linear_solver": "MumpsSolver",
    "tol": 1e-8
}
```

Finally, we can create the solver and obtain the solution for given bounds on the variables and expressions:
```python
ccopt_solver = ca.nlpsol("ccopt_mpcc", "ccopt", nlp, casadi_solver_opts)

res = ccopt_solver(
    x0=x0,
    lbx=lbx,
    ubx=ubx,
    lbg=lbg,
    ubg=ubg,
)

print(f"Solution: x1 = {res['x'][0]} x2 = {res['x'][1]}, x3 = {res['x'][2]}")
```

### Nonlinear complementarity
We now model the more complicated complementarity problem:
```math
  \begin{aligned}
    \min_{x\in\mathbb{R^5}} \; & \frac{1}{2}||x||^2 \\
    \text{subject to} \quad &  0\le x_1 \perp x_2 \ge 0\\
                            &  0\le x_1+2x_3^2 \perp x_5 \ge 0\\
                            &  -2\le x_4 \perp (x_3-5)^2 \ge 0\\
                            &  0\le x_5^3-3x_3^2 \perp x_1^2 + x_2 \ge -3\\
  \end{aligned}
```
As before we first model the expressions and numerical data:
```python
import casadi as ca
import numpy as np

x = ca.SX.sym("x", 5)
x1 = x[0]
x2 = x[1]
x3 = x[2]
x4 = x[3]
x5 = x[4]

g = ca.vertcat(
    x1 + 2*x3**2,
    (x3-5)**2,
    x5**3 - 3*x3**3,
    x1**2 + x2,
)

f = 0.5*ca.norm(x)**2

nlp = {
    "x": x,
    "f": f,
    "g": g,
}

x0 = np.ones(5)
lbx=[0.,0.,-np.inf, -2., 0]
ubx=np.inf*np.ones(5)
lbg=[0., 0., 0., -3.]
ubg=np.inf*np.ones(5)
```
Followed by the complementarity data:
```python
cc_pairs = np.array([[0,1], [0,4], [3,2], [2,3]])
cc_types = np.array([0,2,1,3])

casadi_solver_opts = {
    "cc_pairs": cc_pairs.tolist(),
    "cc_types": cc_types.tolist(),
}
```
Note that here we use all four complementarity types, and index into both `x` and `g`.

Finally, we can again create the solver and obtain the solution for given bounds on the variables and expressions:
```python
ccopt_solver = ca.nlpsol("ccopt_mpcc", "ccopt", nlp, casadi_solver_opts)

res = ccopt_solver(
    x0=x0,
    lbx=lbx,
    ubx=ubx,
    lbg=lbg,
    ubg=ubg,
)

print(f"Solution: x = {res['x']}")
print(f"Solution: g = {res['g']}")
```

## Appendix A: other platforms
### Windows
For windows hosts a similar process can be followed. 
The only major difference is that the shared libraries in windows are stored in the `bin` subdirectory of the libMad installation.
This path needs to then be added to the `PATH`, for example using the steps in [this tutorial](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/).

### MacOS
Installation on `x86-64` and `aarch64` apple platforms follows similarly to the process for linux platforms.
The primary difference involves the use of `DYLD_LIBRARY_PATH` and `DYLD_PRELOAD` for setting up paths and library preload orders.

    !!! warning 
    You will likely have to disable apple "System Integrity Protection" as otherwise, the `DYLD_LIBRARY_PATH` variable will be silently hidden from processes you run. 
    Alternatively you can use `otool` to set absolute paths for `libcasadi_nlpsol_ccopt.dylib` in your virtual environment `site-packages`.
    See this [blog post](https://briandfoy.github.io/macos-s-system-integrity-protection-sanitizes-your-environment/) for more details.
    