import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    """
    Locates files whether running as a script or a frozen EXE.
    PyInstaller unpacks data to sys._MEIPASS.
    """
    if getattr(sys, 'frozen', False):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)

if __name__ == "__main__":
    # 1. Resolve the path to the main application script inside the bundle
    # Note: We bundle bricos_main.py as a data file in the spec.
    app_path = resolve_path("bricos_main.py")
    
    # 2. Construct the system arguments to mimic "streamlit run bricos_main.py"
    # We disable development mode to hide the "Connect to..." menu items.
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
    ]
    
    # 3. Execute Streamlit
    sys.exit(stcli.main())