{
  description = "startml";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;
      in
      {
        # Python and all Python packages are managed by uv (see pyproject.toml / uv.lock);
        # this shell only provides tools and the environment PyPI wheels need on NixOS.
        # Needs `programs.nix-ld.enable = true` in the NixOS config for uv-managed Python.
        devShells.default = pkgs.mkShell {
          name = "dev";

          hardeningDisable = [ "fortify" ];

          packages = with pkgs; [
            uv
            just
            git
            # Large data/model files, see README (DVC is installed as a uv tool instead).
            git-lfs
            rclone
            nvtopPackages.nvidia
          ];

          env = {
            # Triton (torch.compile) looks for libcuda via `/sbin/ldconfig`, absent on NixOS.
            TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
            UV_PYTHON_PREFERENCE = "only-managed";
          };

          shellHook = ''
            # uv-managed Python's OpenSSL doesn't know where NixOS keeps CA certificates.
            # (Set here because `nix develop` doesn't pass SSL_CERT_FILE through `env`.)
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            # NVIDIA driver libraries (libcuda.so) for PyTorch wheels.
            export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        };

        # Supports: nix fmt
        formatter = pkgs.nixfmt-tree;

        # Also supports: nix run .#format
        apps.format = {
          type = "app";
          program = lib.getExe pkgs.nixfmt-tree;
          meta.description = "Format Nix files";
        };
      }
    );
}
