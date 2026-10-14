# How to Run This Project (Terminal Guide)

Step-by-step instructions to launch the Exoplanet Explorer dashboard from a
terminal. Works on Windows, macOS, and Linux.

---

## 1. Prerequisites

- **Python 3.10+** installed (this project was built and tested on Python 3.13).

  Check your version:

  ```bash
  python --version
  # or
  python3 --version
  ```

  If Python is missing, download it from https://www.python.org/downloads/
  (on Windows, tick **"Add python.exe to PATH"** during install).

- **Internet connection** — the app fetches live data from the NASA Exoplanet
  Archive. Without internet it automatically falls back to the saved snapshot
  in `data/snapshot.csv`.

---

## 2. Open a terminal in the project folder

```bash
cd path/to/exoplanet-explorer
```

For example, on this machine:

```bash
cd "C:/Users/rahul.govindu/Desktop/DVST PROJECT/exoplanet-explorer"
```

---

## 3. (Recommended) Create a virtual environment

Keeps the project's packages separate from your global Python.

**Windows (PowerShell / CMD):**

```bash
python -m venv .venv
```

Then activate it:

```bash
# PowerShell
.venv\Scripts\Activate.ps1

# CMD
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt. To deactivate later,
run `deactivate`.

---

## 4. Install the dependencies

```bash
pip install -r requirements.txt
```

This installs: streamlit, pandas, numpy, plotly, requests, statsmodels,
scikit-learn, seaborn, matplotlib, streamlit-autorefresh.

---

## 5. Start the app

```bash
streamlit run app.py
```

You will see output like:

```
Collecting usage statistics...

You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open **http://localhost:8501** in your browser (it usually opens
automatically).

The first load fetches live data from the NASA Exoplanet Archive — this takes
a few seconds. You should see the green banner:

> Data status: **Live** — fetched from the NASA Exoplanet Archive.

If you see "Data status: Offline snapshot" instead, the API was unreachable
and the app is using `data/snapshot.csv` (still fully functional).

---

## 6. Stopping the app

In the terminal running the app, press:

```
Ctrl + C
```

---

## 7. Common commands cheat-sheet

| Task | Command |
|---|---|
| Go to project folder | `cd exoplanet-explorer` |
| Activate virtual environment | `.venv\Scripts\activate` (Windows) / `source .venv/bin/activate` (Mac/Linux) |
| Install dependencies | `pip install -r requirements.txt` |
| Start the dashboard | `streamlit run app.py` |
| Start on a different port | `streamlit run app.py --server.port 8502` |
| Stop the app | `Ctrl + C` |
| Force-refresh data from the sidebar | click **"Refresh data now"** in the app |

---

## 8. Troubleshooting

**`port 8501 is not available` / "Port already in use"**

Another Streamlit process is still running. Kill it:

- Windows (PowerShell):

  ```powershell
  powershell -ExecutionPolicy Bypass -File kill_streamlit.ps1
  ```

- macOS / Linux:

  ```bash
  pkill -f "streamlit run"
  ```

  Then start the app again.

**`streamlit-autorefresh not installed`**

```bash
pip install -r requirements.txt
```

Then restart the app (Ctrl+C, then `streamlit run app.py`).

**Theme looks dark instead of bone-white**

The theme comes from `.streamlit/config.toml`. Make sure you run
`streamlit run app.py` **from inside the `exoplanet-explorer` folder** so
Streamlit finds that file.

**Python / pip not found**

Try `python3` and `pip3` instead of `python` and `pip`:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

**Live fetch fails on first load**

The NASA API may be briefly unreachable — the app falls back to
`data/snapshot.csv` automatically. Click **"Refresh data now"** in the
sidebar to retry the live fetch.

---

## 9. What you should see when it works

- Sidebar: **Controls** (Refresh data now), **Filters** (year range, method,
  planet type, temperature, distance, mission)
- Top banner: **Data status: Live** with a "Last fetched at" timestamp
- KPI row: total planets (filtered), year span, most common method, median
  distance
- Six tabs: **Overview | Discovery Trends | Planet Properties | Forecast |
  Planet Explorer | Data Quality**
- The dashboard auto-refreshes every **10 minutes** while open

For full documentation (features, data source, citations, insights) see
[README.md](README.md).
