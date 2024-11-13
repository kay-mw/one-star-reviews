{ pkgs ? import (fetchTarball
  "https://github.com/NixOS/nixpkgs/archive/862fc0e4ce08540cbd044707d0a8108ed30b3893.tar.gz") {
    overlays = [
      (final: prev: {
        python312 = prev.python312.override {
          packageOverrides = final: prevPy: {
            triton-bin = prevPy.triton-bin.overridePythonAttrs (oldAttrs: {
              postFixup = ''
                chmod +x "$out/${prev.python312.sitePackages}/triton/backends/nvidia/bin/ptxas"
                substituteInPlace $out/${prev.python312.sitePackages}/triton/backends/nvidia/driver.py \
                  --replace \
                    'return [libdevice_dir, *libcuda_dirs()]' \
                    'return [libdevice_dir, "${prev.addDriverRunpath.driverLink}/lib", "${prev.cudaPackages.cuda_cudart}/lib/stubs/"]'
              '';
            });
          };
        };
        python312Packages = final.python312.pkgs;
      })
    ];
  } }:
(pkgs.buildFHSUserEnv {
  name = "brotherman";
  targetPkgs = pkgs:
    with pkgs; [
      # Build essentials
      gcc
      cmake
      gnumake
      binutils
      git # Added git for cloning llama.cpp

      # Original packages
      cudatoolkit
      python312
      # Removed llama-cpp since we'll build from source
      (python312.withPackages
        (python-pkgs: with python-pkgs; [ pip ipython torch-bin ]))
    ];
  multiPkgs = pkgs: with pkgs; [ zlib glibc glibc.dev ];
  runScript = "bash";
  profile = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export EXTRA_CCFLAGS="-I/usr/include"

    export CC=${pkgs.gcc}/bin/gcc
    export CXX=${pkgs.gcc}/bin/g++

    # Remove system llama.cpp from PATH
    export PATH=$(echo $PATH | tr ':' '\n' | grep -v "llama-cpp" | tr '\n' ':')

    source ./tuning/.venv/bin/activate

    zsh
  '';
}).env
