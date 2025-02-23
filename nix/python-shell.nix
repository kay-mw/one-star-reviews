{ pkgs ? import (fetchTarball
  "https://github.com/NixOS/nixpkgs/archive/a3c0b3b21515f74fd2665903d4ce6bc4dc81c77c.tar.gz")
  { }, extraBuildInputs ? [ ], myPython ? pkgs.python3, extraLibPackages ? [ ]
, pythonWithPkgs ? myPython }:

let
  buildInputs = with pkgs;
    [ clang llvmPackages_16.bintools rustup ] ++ extraBuildInputs;

  lib-path = with pkgs; lib.makeLibraryPath buildInputs;

  shell = pkgs.mkShell {
    buildInputs = [
      # my python and packages
      pythonWithPkgs

      # other packages needed for compiling python libs
      pkgs.readline
      pkgs.libffi
      pkgs.openssl

      # unfortunately needed because of messing with LD_LIBRARY_PATH below
      pkgs.git
      pkgs.openssh
      pkgs.rsync

    ] ++ extraBuildInputs;
    shellHook = ''
      # Allow the use of wheels.
      SOURCE_DATE_EPOCH=$(date +%s)

      # Augment the dynamic linker path
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}:${pkgs.stdenv.cc.cc.lib}/lib

      # Setup the virtual environment if it doesn't already exist.
      VENV=.venv
      source ./$VENV/bin/activate
      export PYTHONPATH=$PYTHONPATH:`pwd`/$VENV/${myPython.sitePackages}/

      export SPARK_LOCAL_DIRS=/home/kiran/spark

      zsh
    '';
  };

in shell
