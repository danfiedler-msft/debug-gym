# SFT Data Viewer

A web-based tool for viewing and analyzing Supervised Fine-Tuning (SFT) conversation data in JSONL format.

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python sft_data_viewer.py --safe-root /path/to/sft-data
```

3. Open `http://127.0.0.1:5001` in your browser

4. Upload a JSONL file to view conversation trajectories with:
   - Message-by-message navigation
   - Success/failure indicators  
   - Random shuffle for diverse sampling
   - Dataset statistics and analysis

The viewer binds to `127.0.0.1` and limits **Load from Server** to `.jsonl`
files under the current directory by default. Configure the narrowest suitable
root with `--safe-root` or `SFT_DATA_VIEWER_SAFE_ROOT`; traversal, sibling-prefix
paths, and symlink escapes are rejected. Uploaded files continue to work
independently of the server-side safe root. Use `--host` only when you
intentionally need a non-loopback listener, and add each exact hostname with
`--trusted-host` when serving through a named reverse proxy.

## Data Format

Expects JSONL files with conversation objects containing `messages`, `problem`, `run_id`, `satisfied_criteria`, and token counts.
