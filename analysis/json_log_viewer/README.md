# JSON Log Viewer

A Flask-based web viewer for debug-gym trajectory JSON files. Visualize agent exploration sessions with step-by-step action breakdowns.

## Installation

```bash
cd analysis/json_log_viewer
pip install -r requirements.txt
```

## Usage

Start the server:

```bash
python json_log_viewer.py -p 5050 --safe-root /path/to/trajectories
```

Then open http://127.0.0.1:5050 in your browser.

The viewer binds to `127.0.0.1` and confines server-side browsing and loading to
the current directory by default. Set `--safe-root` (or
`JSON_LOG_VIEWER_SAFE_ROOT`) to the narrowest directory containing the logs you
need. Paths outside that root, including symlink escapes, are rejected. Use
`--host` only when you intentionally need a non-loopback listener.

### Loading Trajectories

You can load trajectory files in several ways:

1. **Upload**: Click "Upload" and select a JSON file
2. **Browse**: Click "Browse Files" to navigate within the configured safe root
3. **API**: Load an in-root file via `GET /load_file_from_path?path=trajectory.json`

### Integration with Gray Tree Frog

Cross-origin requests are disabled by default. To allow Gray Tree Frog's
lineage visualization to open trajectories, configure its exact origin:

```bash
python json_log_viewer.py \
  --safe-root /path/to/trajectories \
  --allowed-origin https://gray-tree-frog.example
```

Repeat `--allowed-origin` for multiple trusted origins, or set a comma-separated
`JSON_LOG_VIEWER_ALLOWED_ORIGINS` value. Wildcard origins are not supported.
The viewer also accepts only loopback `Host` headers by default. If you
intentionally expose it through a named reverse proxy, add each exact hostname
with `--trusted-host`.

## Features

- Step-by-step trajectory visualization
- Color-coded action types (bash, view, edit, etc.)
- Detailed bash command classification
- Statistics view showing action distribution
- Keyboard navigation between steps
