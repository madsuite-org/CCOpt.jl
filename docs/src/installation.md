# Installation
To install CCOpt for use in `Julia`, simply proceed to
```julia
pkg> add CCOpt
```

## Installing `libMad`
To install `CCOpt` for use outside of `Julia` you will need to install the `libMad` shared library bundle.
The library can be installed from [its GitHub releases page](https://github.com/madsuite-org/libMad/releases/latest) or by the following shell commands:
```shell
curl -L https://github.com/madsuite-org/libMad/releases/download/<release-tag>/libMad-ubuntu-x86_64-<release-tag>.tar.gz --output libMad.tar.gz
tar -xzf libMad.tar.gz
```
in the location you wish to install the library to.
This downloads the bundle and untars its contents

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

