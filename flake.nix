{
  description = "Jupyter development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems =
        [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f:
        nixpkgs.lib.genAttrs supportedSystems
          (system: f { pkgs = import nixpkgs { inherit system; }; });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs;
            [ python312 fontconfig mesa xcb-util-cursor libGL adwaita-icon-theme libsForQt5.qt5.qtwayland napari ]
            ++ (with pkgs.python312Packages; [
              pip
              icecream
              virtualenv
              black
              isort

              ipykernel
              jupyterlab
              jupyter
              ipywidgets

              tqdm
              # numpy
              # matplotlib

              # scikit-learn
              pandas
              seaborn

              python-fontconfig

              # pytorch
              # monai

              # lime

              # torchvision
              wget
              # opencv-python

              # torchtext

              # captum

              # adversarial-robustness-toolbox
              pyqt5
            ]);
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.fontconfig.lib}/lib:$LD_LIBRARY_PATH;
            export LD_LIBRARY_PATH=${pkgs.mesa.out}/lib:$LD_LIBRARY_PATH;
            export LD_LIBRARY_PATH=${pkgs.xcb-util-cursor}/lib:$LD_LIBRARY_PATH;
            export LD_LIBRARY_PATH=${pkgs.libGL}/lib:$LD_LIBRARY_PATH;
            export LD_LIBRARY_PATH=${pkgs.libglvnd}/lib:$LD_LIBRARY_PATH;
            export LD_LIBRARY_PATH=${pkgs.qt5.qtbase}/lib:$LD_LIBRARY_PATH;
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH;
            export QT_QPA_PLATFORM_PLUGIN_PATH=${pkgs.qt5.qtbase}/lib/qt-${pkgs.qt5.qtbase.version}/plugins/platforms;
            export QT_PLUGIN_PATH=${pkgs.qt5.qtbase}/lib/qt/plugins;
            export XDG_DATA_DIRS=${pkgs.adwaita-icon-theme}/share:$XDG_DATA_DIRS;
            export QT_QPA_PLATFORM=wayland;
          '';
        };
      });
    };
}
