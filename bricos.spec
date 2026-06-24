# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build recipe for BriCoS (one-folder / "onedir" build).

Build it with:   pyinstaller bricos.spec
(or run build.ps1, which installs the dependencies first). See PACKAGING.md.

The result is dist/BriCoS/ containing BriCoS.exe plus an _internal/ folder.
Distribute the WHOLE folder (zip it). The entry point is run_app.py, which
launches Streamlit on bricos_main.py and prepares a persistent numba cache so
the JIT compile happens once, not on every launch.
"""
from PyInstaller.utils.hooks import (
    collect_all, collect_data_files, collect_submodules, copy_metadata,
)

# --- BriCoS's own files ----------------------------------------------------
# Bundled at the bundle root (sys._MEIPASS). Streamlit runs bricos_main.py from
# there, so its sibling "import bricos_*" resolve from the same directory, and
# resource_path()/resolve_path() find the data files there.
app_modules = [
    'bricos_main.py', 'bricos_data.py', 'bricos_solver.py', 'bricos_kernels.py',
    'bricos_viz.py', 'bricos_report.py', 'bricos_results_ui.py',
    'bricos_export.py', 'bricos_manual.py',
]
app_data = ['vehicles.csv', 'logo.png']
datas = [(f, '.') for f in app_modules + app_data]
binaries = []
hiddenimports = []

# --- Streamlit -------------------------------------------------------------
# No PyInstaller hook ships for Streamlit, so collect it explicitly: its
# package data (the built frontend in streamlit/static, runtime assets), its
# submodules (many are imported dynamically by name), and the importlib
# metadata it reads to check its own and its dependencies' versions.
datas += collect_data_files('streamlit')
hiddenimports += collect_submodules('streamlit')
for pkg in ('streamlit', 'numpy', 'pandas', 'plotly', 'numba', 'reportlab',
            'xlsxwriter', 'kaleido', 'altair', 'pyarrow', 'packaging',
            'tornado', 'click', 'rich', 'tenacity', 'toml', 'watchdog',
            'gitpython', 'pympler'):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# --- App runtime dependencies ----------------------------------------------
# CRITICAL: the bricos_*.py modules are run by Streamlit from data files, so
# PyInstaller's analysis of run_app.py (which imports only streamlit) never
# sees THEIR third-party imports. Each runtime dependency must therefore be
# bundled explicitly with collect_all (modules + data + binaries), or it will
# be missing from the package and fail only at runtime. Covers:
#   plotly (diagrams), reportlab (PDF report + manual), xlsxwriter (Excel
#   export), numba (+ the bundled LLVM runtime), numpy/pandas, and
#   kaleido + choreographer for PDF figure export. Kaleido 1.x does NOT bundle
#   a browser - it drives the system Chrome/Edge at runtime (Windows ships
#   Edge) - so no browser is bundled, only the Python packages.
for pkg in ('plotly', 'reportlab', 'xlsxwriter', 'numba', 'numpy', 'pandas',
            'kaleido', 'choreographer', 'logistro'):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass


a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BriCoS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,           # show the console so the local URL and any errors are visible (beta)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='BriCoS',
)
