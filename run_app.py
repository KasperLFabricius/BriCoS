import streamlit.web.cli as stcli
import os, sys, shutil

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

def prepare_numba_cache():
    """Give numba a stable source path and cache dir across EXE launches.

    PyInstaller (onefile) extracts the bundle to a fresh temp dir on every
    launch. Numba keys its disk cache by a hash of the source file's
    DIRECTORY (numba caching.py, get_suitable_cache_subpath), so kernels
    imported from the extraction dir can never hit a previous launch's
    cache and every start re-JIT-compiles all kernels - tens of seconds.

    Fix: copy the kernels module to a stable per-user dir and put it first
    on sys.path, so numba sees the same source path every launch. The
    cache freshness stamp in frozen mode is the executable's stat (numba
    _SourceFileBackedLocatorMixin.get_source_stamp), so the cache
    invalidates exactly when the EXE is rebuilt. NUMBA_CACHE_DIR keeps the
    compiled artifacts in the same per-user tree.

    Caching is an optimization: any failure must never block startup.
    """
    if not getattr(sys, 'frozen', False):
        return None
    try:
        appdata = os.environ.get('APPDATA') or os.path.expanduser('~')
        base_dir = os.path.join(appdata, 'BriCoS')
        stable_src = os.path.join(base_dir, 'numba_src')
        os.makedirs(stable_src, exist_ok=True)
        shutil.copy2(resolve_path('bricos_kernels.py'), stable_src)
        sys.path.insert(0, stable_src)
        os.environ.setdefault('NUMBA_CACHE_DIR',
                              os.path.join(base_dir, 'numba_cache'))
        return stable_src
    except Exception:
        return None

APP_PORT = 8501

if __name__ == "__main__":
    # 0. Persistent numba cache (frozen builds re-JIT every launch otherwise)
    prepare_numba_cache()

    # No anonymous usage statistics from a distributed build.
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    # 1. Resolve the path to the main application script inside the bundle
    # Note: We bundle bricos_main.py as a data file in the spec.
    app_path = resolve_path("bricos_main.py")

    # 2. Construct the system arguments to mimic "streamlit run bricos_main.py".
    # developmentMode=false hides the "Connect to..." menu items. headless=true
    # is essential for a packaged build: it skips Streamlit's first-run e-mail
    # prompt (which reads from the console and would otherwise hang a double-
    # clicked EXE) and Streamlit's own browser launch - we open the browser
    # ourselves below.
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.headless=true",
        f"--server.port={APP_PORT}",
    ]

    # 3. Open the browser ourselves (headless mode does not). A short delay lets
    # the server bind first; any failure here must never stop the app.
    try:
        import threading
        import webbrowser
        threading.Timer(2.5, lambda: webbrowser.open(f"http://localhost:{APP_PORT}")).start()
    except Exception:
        pass

    # 4. Execute Streamlit
    sys.exit(stcli.main())
