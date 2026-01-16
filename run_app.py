import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    """
    Locates the file in either the source directory (development)
    or the PyInstaller temporary _MEI folder (runtime).
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, path)

if __name__ == "__main__":
    # 1. Locate the main Streamlit file
    main_script = resolve_path("bricos_main.py")

    # 2. Modify sys.argv to emulate "streamlit run bricos_main.py"
    # The first argument is the script name (streamlit), the rest are args
    sys.argv = [
        "streamlit",
        "run",
        main_script,
        "--global.developmentMode=false",
        "--server.headless=true",  # Don't show server dialogs
    ]

    # 3. Launch Streamlit
    print("Launching BriCoS v0.30...")
    sys.exit(stcli.main())