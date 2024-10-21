# Run with `nix-shell cuda-fhs.nix`
{ pkgs ? import <nixpkgs> { } }:
(pkgs.buildFHSUserEnv {
  name = "cuda-env";
  targetPkgs = pkgs:
    with pkgs; [
      cudatoolkit
      python312
      python312Packages.pip
      python312Packages.ipython
      python312Packages.virtualenv
    ];
  multiPkgs = pkgs: with pkgs; [ zlib ];
  runScript = "bash";
  profile = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
    # export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
    zsh

    VENV=.venv
    if test ! -d $VENV; then
      virtualenv $VENV
    fi
    source ./$VENV/bin/activate
  '';
}).env
