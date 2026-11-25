{
  description = "Vestige: Deterministic local Rust Causal Proof Engine and operating system for AI coding agents";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        vestigePkg = pkgs.rustPlatform.buildRustPackage {
          pname = "vestige";
          version = "7.0.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
          cargoBuildFlags = [ "-p" "vestige-mcp" "--bin" "vestige" ];
          doCheck = false;
          meta = with pkgs.lib; {
            description = "Deterministic Rust causal proof engine, zero-cloud GitHub PR bots, and fail-closed agent Execution Leases";
            homepage = "https://github.com/samvallad33/vestige";
            license = licenses.agpl3Only;
            mainProgram = "vestige";
          };
        };
      in
      {
        packages = {
          default = vestigePkg;
          vestige = vestigePkg;
        };

        apps.default = {
          type = "app";
          program = "${vestigePkg}/bin/vestige";
        };
      }
    ) // {
      homeManagerModules.default = { config, lib, pkgs, ... }:
        let
          cfg = config.programs.vestige;
        in
        {
          options.programs.vestige = {
            enable = lib.mkEnableOption "Vestige deterministic causal proof engine";
          };
          config = lib.mkIf cfg.enable {
            home.packages = [ self.packages.${pkgs.system}.default ];
          };
        };
    };
}
