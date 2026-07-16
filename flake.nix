{
  description = "universal-llm";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    sse-parser-src = {
      url = "github:n3wm1nd/runix-project?dir=libs/sse-parser";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, sse-parser-src, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      haskellPackages = pkgs.haskellPackages.override {
        overrides = self: super: {
          sse-parser = self.callCabal2nix "sse-parser" sse-parser-src { };
        };
      };

      universal-llm = haskellPackages.developPackage {
        name = "universal-llm";
        root = ./.;
      };
    in
    {
      packages.${system} = {
        default = universal-llm;
        universal-llm = universal-llm;
      };

      devShells.${system}.default = haskellPackages.shellFor {
        packages = p: [ universal-llm ];
        buildInputs = [ pkgs.cabal-install ];
      };
    };
}
