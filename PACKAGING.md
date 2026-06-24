# Packaging BriCoS as a Windows executable

This guide turns the Python project into a standalone Windows app you can hand
to someone who does **not** have Python installed. It is written for a first-
time packager - just follow the steps in order.

The tool used is **PyInstaller**. It reads the recipe in [`bricos.spec`](bricos.spec)
and produces a folder you can zip and share.

---

## What you get

A folder: `dist\BriCoS\`

- `BriCoS.exe` - double-click to start the app (it opens in your browser).
- `_internal\` - all the libraries and data the app needs.

You distribute the **whole `dist\BriCoS` folder** (zipped). The recipient
unzips it anywhere and runs `BriCoS.exe`. **No Python install needed.**

This is a *one-folder* build (chosen for reliability and faster startup). It is
not a single `.exe`; that is normal and expected.

---

## Before you start (one-time)

1. Install **Python 3.10+** (3.13 is what this project is tested on) from
   <https://www.python.org/downloads/> - tick **"Add Python to PATH"** during
   install.
2. Get the BriCoS source code (this repository) onto your machine.
3. Open a terminal (PowerShell or Command Prompt) **in the project folder**
   (the folder that contains `bricos.spec`).

You do not need to install anything else by hand - the build script does it.

---

## Build it (the easy way)

**Double-click `build.bat`** (or run `build.bat` in the terminal).

It will:
1. install the app's dependencies (`requirements.txt`),
2. install PyInstaller,
3. build the app.

The first build takes a few minutes and prints a lot of text - that is normal.
When it finishes you will see:

```
DONE. Your app is in:  dist\BriCoS\
```

### Build it (the manual way)

If you prefer to run the commands yourself:

```powershell
python -m pip install -r requirements.txt
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean bricos.spec
```

---

## Run / test it

Double-click `dist\BriCoS\BriCoS.exe`.

- A small **console window** opens and prints a local web address
  (`http://localhost:8501`). Keep this window open - closing it stops the app.
- Your **browser opens automatically** to the app. (If it doesn't, copy the
  address from the console into your browser.)
- The **first time you run it**, the app spends ~10-40 s compiling its math
  kernels. This is a one-time cost: the result is cached in
  `%APPDATA%\BriCoS\`, so **every later launch starts quickly**. (Re-building
  the .exe resets this cache, so the first run after a new build is slow again.)

Quick things to check the build is healthy:
- the four result diagrams appear,
- **Generate PDF report** and the manual's **Download PDF** produce a file
  (these need a Chromium browser on the machine - Windows ships **Edge**, so
  this works out of the box),
- the **Tabular Data** Excel/CSV downloads work.

---

## Share it

Zip the whole `dist\BriCoS` folder and send it. The recipient:
1. unzips it,
2. double-clicks `BriCoS.exe`.

Requirements on their machine: **64-bit Windows 10/11** and a **Chromium-based
browser** (Microsoft Edge is pre-installed on Windows, so this is satisfied by
default - it is only needed for the PDF export).

---

## How it works (for the curious)

- **`run_app.py`** is the entry point. When frozen it launches Streamlit on
  `bricos_main.py`, runs **headless** (so it doesn't show Streamlit's first-run
  e-mail prompt, which would otherwise freeze a double-clicked .exe) and opens
  your browser itself.
- It also copies the numba kernel module to `%APPDATA%\BriCoS` and points the
  numba cache there, so the heavy just-in-time compile happens **once** and is
  reused across launches - that is what preserves calculation speed.
- **`bricos.spec`** tells PyInstaller what to bundle. The important, non-obvious
  part: Streamlit runs `bricos_main.py` as a *script* loaded from a data file,
  so PyInstaller cannot automatically discover the app's third-party imports
  (reportlab, plotly, kaleido, xlsxwriter, numba, ...). The spec therefore
  bundles each of them explicitly with `collect_all`. If you add a **new
  third-party dependency**, add it both to `requirements.txt` and to the
  `collect_all` list in `bricos.spec`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'python' is not recognized` | Python isn't on PATH. Reinstall Python and tick "Add Python to PATH", or use the full path to `python.exe`. |
| Build fails on `pip install` | Check your internet connection; re-run `build.bat`. |
| The .exe opens a console then closes instantly | Run it from a terminal so you can read the error, or check that port 8501 isn't already in use. |
| App starts but `No module named '<x>'` | A dependency wasn't bundled. Add `<x>` to the `collect_all` list in `bricos.spec` and rebuild. |
| "Generate PDF" produces no figures | No Chromium browser found. Install Microsoft Edge or Google Chrome on that machine. |
| First launch is slow every time | The cache in `%APPDATA%\BriCoS` isn't persisting (e.g. it's cleared on logout). It is rebuilt automatically; this only affects the first run after each rebuild. |

If a build error mentions a specific package, the usual fix is to add that
package to the `collect_all` loop in `bricos.spec` and rebuild.
