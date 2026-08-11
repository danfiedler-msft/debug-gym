import argparse
import json
import logging
import math
import os
import stat
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5001
DEFAULT_SAFE_ROOT = Path.cwd().resolve()
DEFAULT_TRUSTED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
ALLOWED_SUFFIXES = {".jsonl"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = (
    5 * 1024 * 1024 * 1024
)  # 5GB max file size for large JSONL files
app.config["SAFE_ROOT"] = (
    Path(os.environ.get("SFT_DATA_VIEWER_SAFE_ROOT", DEFAULT_SAFE_ROOT))
    .expanduser()
    .resolve()
)
app.config["TRUSTED_HOSTS"] = list(DEFAULT_TRUSTED_HOSTS)

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global variables to store loaded data info
current_file_path = None
current_file_name = None
current_file_descriptor = None
current_file_lock = threading.Lock()
total_records = 0
records_per_page = 10


class SafePathError(ValueError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def resolve_safe_path(
    raw_path: str,
    *,
    allowed_suffixes: set[str] | None = None,
    expect: str | None = None,
) -> Path:
    """Resolve a user path canonically and confine it to the configured root."""
    if not raw_path:
        raise SafePathError("A path is required", 400)

    try:
        safe_root = Path(app.config["SAFE_ROOT"]).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SafePathError("The configured safe root is unavailable", 500) from exc

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = safe_root / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SafePathError("Invalid path", 400) from exc

    try:
        resolved.relative_to(safe_root)
    except ValueError as exc:
        raise SafePathError("Access denied", 403) from exc

    if expect == "file" and not resolved.is_file():
        raise SafePathError("File not found", 404)
    if allowed_suffixes is not None and resolved.suffix.lower() not in allowed_suffixes:
        raise SafePathError("Invalid file type", 400)
    return resolved


def open_confined_server_file(filepath: Path) -> int:
    """Open a validated file and pin the inode used by later requests."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    descriptor = os.open(filepath, flags)
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise SafePathError("File not found", 404)
        if descriptor_stat.st_nlink != 1:
            raise SafePathError("Files with multiple hard links are not allowed", 403)

        revalidated_path = resolve_safe_path(
            str(filepath),
            allowed_suffixes=ALLOWED_SUFFIXES,
            expect="file",
        )
        path_stat = os.stat(revalidated_path, follow_symlinks=False)
        if revalidated_path != filepath or (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ) != (path_stat.st_dev, path_stat.st_ino):
            raise SafePathError("File changed while it was being loaded", 409)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def save_uploaded_file(file_storage) -> tuple[int, Path, str]:
    """Save an upload through an exclusive descriptor with a generated name."""
    filename = secure_filename(file_storage.filename)
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise SafePathError("Please upload a JSONL file", 400)

    configured_root = Path(app.config["UPLOAD_FOLDER"]).absolute()
    if configured_root.is_symlink():
        raise OSError("Upload folder must not be a symbolic link")
    configured_root.mkdir(parents=True, exist_ok=True)
    upload_root = configured_root.resolve(strict=True)
    filepath = upload_root / f"{uuid.uuid4().hex}.jsonl"
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    descriptor = os.open(filepath, flags, 0o600)
    try:
        with os.fdopen(os.dup(descriptor), "wb") as destination:
            file_storage.save(destination)
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode) or descriptor_stat.st_nlink != 1:
            raise OSError("Upload destination is not a regular file")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, filepath, filename
    except Exception:
        os.close(descriptor)
        filepath.unlink(missing_ok=True)
        raise


@contextmanager
def open_descriptor(descriptor: int):
    duplicate = os.dup(descriptor)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with os.fdopen(duplicate, "r", encoding="utf-8") as file:
            duplicate = None
            yield file
    finally:
        if duplicate is not None:
            os.close(duplicate)


@contextmanager
def open_current_file():
    with current_file_lock:
        if current_file_descriptor is None:
            raise RuntimeError("No file loaded")
        with open_descriptor(current_file_descriptor) as file:
            yield file


def replace_current_file(descriptor: int, filepath: Path, filename: str) -> None:
    global current_file_descriptor, current_file_name, current_file_path
    with current_file_lock:
        previous_descriptor = current_file_descriptor
        current_file_descriptor = descriptor
        current_file_path = filepath
        current_file_name = filename
        if previous_descriptor is not None:
            os.close(previous_descriptor)


def clear_current_file() -> None:
    global current_file_descriptor, current_file_name, current_file_path
    with current_file_lock:
        if current_file_descriptor is not None:
            os.close(current_file_descriptor)
        current_file_descriptor = None
        current_file_path = None
        current_file_name = None


def to_pretty_json(value):
    """Convert Python object to pretty-printed JSON string with minimal whitespace"""
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).strip()


# Add custom filters and globals to Jinja2
app.jinja_env.filters["tojson_pretty"] = to_pretty_json
app.jinja_env.globals.update(min=min, max=max)


def count_jsonl_lines(descriptor):
    """Count total lines in JSONL file efficiently"""
    with open_descriptor(descriptor) as f:
        return sum(1 for _ in f)


def load_jsonl_page(page=0, per_page=10):
    """Load a specific page of records from JSONL file"""
    records = []
    start_idx = page * per_page
    end_idx = start_idx + per_page

    try:
        with open_current_file() as f:
            for i, line in enumerate(f):
                if i >= end_idx:
                    break
                if i >= start_idx:
                    try:
                        record = json.loads(line.strip())
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i}: {e}")
                        continue
        return records
    except Exception as e:
        print(f"Error loading page: {e}")
        return []


def get_shuffled_indices(total_records, page=0, per_page=10, seed=None):
    """Generate shuffled indices for a specific page"""
    import random

    if seed is not None:
        random.seed(seed)

    # Create a list of all indices and shuffle them
    all_indices = list(range(total_records))
    random.shuffle(all_indices)

    # Get the requested page of shuffled indices
    start_idx = page * per_page
    end_idx = start_idx + per_page
    return all_indices[start_idx:end_idx]


def load_records_by_indices(indices):
    """Load specific records by their line indices"""
    records = []
    indices_set = set(indices)

    try:
        with open_current_file() as f:
            for i, line in enumerate(f):
                if i in indices_set:
                    try:
                        record = json.loads(line.strip())
                        records.append((i, record))  # Keep original index
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i}: {e}")
                        continue

                # Early exit if we've found all records we need
                if len(records) == len(indices):
                    break
    except Exception as e:
        print(f"Error loading records by indices: {e}")
        return []

    # Sort records to match the order of indices
    index_to_record = {idx: record for idx, record in records}
    return [
        (idx, index_to_record.get(idx)) for idx in indices if idx in index_to_record
    ]


def load_single_record(record_idx):
    """Load a single record by index from JSONL file"""
    try:
        with open_current_file() as f:
            for i, line in enumerate(f):
                if i == record_idx:
                    try:
                        return json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"Error parsing record {record_idx}: {e}")
                        return None
        return None
    except Exception as e:
        print(f"Error loading record: {e}")
        return None


@app.route("/")
def index():
    global current_file_name, total_records

    if current_file_descriptor is None:
        return redirect(url_for("file_upload"))

    # Get pagination and shuffle parameters
    page = request.args.get("page", 0, type=int)
    shuffle = request.args.get("shuffle", "false").lower() == "true"

    # Calculate pagination info
    total_pages = math.ceil(total_records / records_per_page)

    if shuffle:
        # Generate shuffled indices for this page
        shuffled_indices = get_shuffled_indices(total_records, page, records_per_page)
        # Load records by their shuffled indices
        indexed_records = load_records_by_indices(shuffled_indices)
        records = [(idx, record) for idx, record in indexed_records]
    else:
        # Load records normally
        normal_records = load_jsonl_page(page, records_per_page)
        records = [
            (page * records_per_page + i, record)
            for i, record in enumerate(normal_records)
        ]

    # Process records for display (extract key info)
    processed_records = []
    for original_idx, record in records:
        record_idx = original_idx

        # Extract key metadata
        satisfied_criteria = record.get("satisfied_criteria", False)
        has_satisfied_criteria = (
            len(satisfied_criteria) > 0
            if isinstance(satisfied_criteria, list)
            else bool(satisfied_criteria)
        )
        criteria_count = (
            len(satisfied_criteria)
            if isinstance(satisfied_criteria, list)
            else (1 if satisfied_criteria else 0)
        )

        metadata = {
            "index": record_idx,
            "problem": record.get("problem", "N/A"),
            "run_id": record.get("run_id", "N/A"),
            "satisfied_criteria": satisfied_criteria,
            "has_satisfied_criteria": has_satisfied_criteria,
            "criteria_count": criteria_count,
            "truncated": record.get("truncated", False),
            "tokens": record.get("#tokens", 0),
            "messages_count": len(record.get("messages", [])),
            "tools_count": len(record.get("tools", [])) if record.get("tools") else 0,
        }

        # Extract conversation preview (first few messages)
        messages = record.get("messages", [])
        conversation_preview = []
        for msg in messages[:3]:  # Show first 3 messages as preview
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate content for preview
            if len(content) > 200:
                content = content[:200] + "..."
            conversation_preview.append({"role": role, "content": content})

        processed_records.append(
            {
                "metadata": metadata,
                "conversation_preview": conversation_preview,
                "has_more_messages": len(messages) > 3,
            }
        )

    return render_template(
        "index.html",
        processed_records=processed_records,
        current_page=page,
        total_pages=total_pages,
        total_records=total_records,
        current_file=current_file_name,
        records_per_page=records_per_page,
        is_shuffled=shuffle,
    )


@app.route("/upload", methods=["GET", "POST"])
def file_upload():
    global total_records

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", error="No file selected")

        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", error="No file selected")

        if file:
            descriptor = None
            filepath = None

            try:
                descriptor, filepath, filename = save_uploaded_file(file)
                total_records = count_jsonl_lines(descriptor)
                replace_current_file(descriptor, filepath, filename)
                descriptor = None
                return redirect(url_for("index"))
            except SafePathError as exc:
                return render_template("upload.html", error=str(exc)), exc.status_code
            except (OSError, UnicodeError):
                if descriptor is not None:
                    os.close(descriptor)
                # Clean up the file if it was partially saved
                if filepath is not None:
                    filepath.unlink(missing_ok=True)
                logging.exception("Failed to load uploaded SFT data")
                return render_template(
                    "upload.html",
                    error="Unable to load file. For large files, use the "
                    "'Load from Server' option instead.",
                )
        else:
            return render_template("upload.html", error="Please upload a JSONL file")

    return render_template("upload.html")


@app.route("/load_file", methods=["POST"])
def load_file():
    """Load a file from the server filesystem"""
    global total_records

    descriptor = None
    try:
        filepath = resolve_safe_path(
            request.form.get("filepath", "").strip(),
            allowed_suffixes=ALLOWED_SUFFIXES,
            expect="file",
        )
        descriptor = open_confined_server_file(filepath)
        total_records = count_jsonl_lines(descriptor)
        replace_current_file(descriptor, filepath, filepath.name)
        descriptor = None
        return redirect(url_for("index"))
    except SafePathError as exc:
        return render_template("upload.html", error=str(exc)), exc.status_code
    except (OSError, UnicodeError):
        logging.exception("Failed to load SFT data")
        return render_template("upload.html", error="Unable to load file"), 500
    finally:
        if descriptor is not None:
            os.close(descriptor)


@app.route("/record/<int:record_idx>")
def view_record(record_idx):
    """View a single record in detail"""
    if current_file_descriptor is None:
        return redirect(url_for("file_upload"))

    if record_idx < 0 or record_idx >= total_records:
        return jsonify({"error": "Record not found"}), 404

    record = load_single_record(record_idx)
    if record is None:
        return jsonify({"error": "Record not found"}), 404

    # Extract metadata
    satisfied_criteria = record.get("satisfied_criteria", [])
    metadata = {
        "index": record_idx,
        "problem": record.get("problem", "N/A"),
        "run_id": record.get("run_id", "N/A"),
        "satisfied_criteria": satisfied_criteria,
        "satisfied_criteria_list": (
            satisfied_criteria if isinstance(satisfied_criteria, list) else []
        ),
        "has_satisfied_criteria": (
            len(satisfied_criteria) > 0
            if isinstance(satisfied_criteria, list)
            else bool(satisfied_criteria)
        ),
        "truncated": record.get("truncated", False),
        "tokens": record.get("#tokens", 0),
        "messages_count": len(record.get("messages", [])),
        "tools_count": len(record.get("tools", [])) if record.get("tools") else 0,
    }

    # Process messages for better display
    messages = record.get("messages", [])
    tools = record.get("tools", [])

    # No need to process tool calls - we'll handle JSON formatting in the frontend

    return render_template(
        "record_detail.html",
        record=record,
        metadata=metadata,
        messages=messages,
        tools=tools,
        record_idx=record_idx,
        total_records=total_records,
        current_file=current_file_name,
    )


@app.route("/api/record/<int:record_idx>")
def get_record_api(record_idx):
    """API endpoint to get record data as JSON"""
    if current_file_descriptor is None:
        return jsonify({"error": "No file loaded"}), 400

    if record_idx < 0 or record_idx >= total_records:
        return jsonify({"error": "Record not found"}), 404

    record = load_single_record(record_idx)
    if record is None:
        return jsonify({"error": "Record not found"}), 404

    return jsonify(record)


@app.route("/statistics")
def statistics():
    """Show statistics about the loaded dataset"""
    global total_records

    if current_file_descriptor is None:
        return redirect(url_for("file_upload"))

    # Decide whether to sample or use all records based on dataset size
    use_all_records = total_records <= 2000  # Use all records for datasets under 2k

    if use_all_records:
        # Load all records for accurate statistics
        sample_records = []
        with open_current_file() as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    sample_records.append(record)
                except json.JSONDecodeError:
                    continue
        sample_size = len(sample_records)
    else:
        # For larger datasets, sample evenly distributed records
        sample_size = 2000  # Sample 2k records for better performance

        # Create evenly distributed sample indices
        step = total_records / sample_size
        sample_indices = [int(i * step) for i in range(sample_size)]

        sample_records = []
        with open_current_file() as f:
            for i, line in enumerate(f):
                if i in sample_indices:
                    try:
                        record = json.loads(line.strip())
                        sample_records.append(record)
                    except json.JSONDecodeError:
                        continue
        sample_size = len(sample_records)

    # Collect statistics
    stats = {
        "total_records": total_records,
        "sample_size": sample_size,
        "using_all_records": use_all_records,
        "message_counts": [],
        "token_counts": [],
        "satisfied_criteria_count": 0,
        "truncated_count": 0,
        "role_counts": {},
        "criteria_distribution": {},  # Track different criteria combinations
        "all_criteria": set(),  # Track all unique criteria seen
        "criteria_counts": {},  # Count how often each criterion appears
    }

    for record in sample_records:
        # Message and token counts
        messages = record.get("messages", [])
        stats["message_counts"].append(len(messages))
        stats["token_counts"].append(record.get("#tokens", 0))

        # Criteria and truncation analysis
        satisfied_criteria = record.get("satisfied_criteria", [])
        if isinstance(satisfied_criteria, list):
            # Track criteria combinations
            criteria_key = (
                tuple(sorted(satisfied_criteria))
                if satisfied_criteria
                else ("no_criteria",)
            )
            stats["criteria_distribution"][criteria_key] = (
                stats["criteria_distribution"].get(criteria_key, 0) + 1
            )

            # Track all unique criteria
            stats["all_criteria"].update(satisfied_criteria)

            # Count individual criteria occurrences
            for criterion in satisfied_criteria:
                stats["criteria_counts"][criterion] = (
                    stats["criteria_counts"].get(criterion, 0) + 1
                )

            # Count as successful if there are satisfied criteria (can be refined later)
            if len(satisfied_criteria) > 0:
                stats["satisfied_criteria_count"] += 1
        elif satisfied_criteria:  # Handle boolean case for backward compatibility
            stats["satisfied_criteria_count"] += 1
            stats["criteria_distribution"][("legacy_boolean",)] = (
                stats["criteria_distribution"].get(("legacy_boolean",), 0) + 1
            )
        if record.get("truncated", False):
            stats["truncated_count"] += 1

        # Role statistics
        for msg in messages:
            role = msg.get("role", "unknown")
            stats["role_counts"][role] = stats["role_counts"].get(role, 0) + 1

    # Calculate averages and percentages
    if stats["message_counts"]:
        stats["avg_messages"] = sum(stats["message_counts"]) / len(
            stats["message_counts"]
        )
        stats["max_messages"] = max(stats["message_counts"])
        stats["min_messages"] = min(stats["message_counts"])

    if stats["token_counts"]:
        stats["avg_tokens"] = sum(stats["token_counts"]) / len(stats["token_counts"])
        stats["max_tokens"] = max(stats["token_counts"])
        stats["min_tokens"] = min(stats["token_counts"])

    stats["satisfied_criteria_percent"] = (
        (stats["satisfied_criteria_count"] / sample_size * 100)
        if sample_size > 0
        else 0
    )
    stats["truncated_percent"] = (
        (stats["truncated_count"] / sample_size * 100) if sample_size > 0 else 0
    )

    # Process criteria statistics for template
    stats["all_criteria_list"] = sorted(list(stats["all_criteria"]))
    stats["criteria_combinations"] = []
    for criteria_tuple, count in sorted(
        stats["criteria_distribution"].items(), key=lambda x: x[1], reverse=True
    ):
        criteria_list = list(criteria_tuple)
        percentage = (count / sample_size * 100) if sample_size > 0 else 0
        stats["criteria_combinations"].append(
            {
                "criteria": criteria_list,
                "count": count,
                "percentage": percentage,
                "criteria_text": (
                    ", ".join(criteria_list)
                    if criteria_list != ["no_criteria"]
                    else "No criteria satisfied"
                ),
            }
        )

    # Individual criteria statistics
    stats["individual_criteria"] = []
    for criterion, count in sorted(
        stats["criteria_counts"].items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / sample_size * 100) if sample_size > 0 else 0
        stats["individual_criteria"].append(
            {"name": criterion, "count": count, "percentage": percentage}
        )

    return render_template(
        "statistics.html", stats=stats, current_file=current_file_name
    )


@app.route("/change_file")
def change_file():
    clear_current_file()
    return redirect(url_for("file_upload"))


@app.errorhandler(413)
def too_large(e):
    return (
        render_template(
            "upload.html",
            error="File too large! The uploaded file exceeds the maximum size limit. Please use the 'Load from Server' option for large files.",
        ),
        413,
    )


def main():
    parser = argparse.ArgumentParser(description="View SFT JSONL data")
    parser.add_argument(
        "--host",
        default=os.environ.get("SFT_DATA_VIEWER_HOST", DEFAULT_HOST),
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=int(os.environ.get("SFT_DATA_VIEWER_PORT", DEFAULT_PORT)),
    )
    parser.add_argument(
        "--safe-root",
        type=Path,
        default=app.config["SAFE_ROOT"],
        help="Only JSONL files below this directory may be loaded from the server",
    )
    parser.add_argument(
        "--trusted-host",
        action="append",
        help="Exact Host header accepted by the viewer",
    )
    args = parser.parse_args()

    try:
        app.config["SAFE_ROOT"] = args.safe_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        parser.error(f"safe root is unavailable: {exc}")
    if args.trusted_host is not None:
        app.config["TRUSTED_HOSTS"] = args.trusted_host
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
