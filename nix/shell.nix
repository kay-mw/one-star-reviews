let
  pkgs = import (fetchTarball
    "https://github.com/NixOS/nixpkgs/archive/a3c0b3b21515f74fd2665903d4ce6bc4dc81c77c.tar.gz")
    { };
  myPython = pkgs.python311;
  pythonPackages = pkgs.python311Packages;

  pythonWithPkgs = myPython.withPackages (pythonPkgs:
    with pythonPkgs; [
      ipython
      pip
      debugpy
      setuptools
      virtualenvwrapper
      wheel
      grpcio
    ]);

  extraBuildInputs = with pythonPackages;
    [ ] ++ (with pkgs; [
      jdk # Pyspark
      duckdb
    ]);
in import ./python-shell.nix {
  extraBuildInputs = extraBuildInputs;
  # extraLibPackages = extraLibPackages;
  myPython = myPython;
  pythonWithPkgs = pythonWithPkgs;
  pkgs = pkgs;
}
