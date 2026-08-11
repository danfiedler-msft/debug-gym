import os
import re

import pytest

from debug_gym.gym.terminals.local import LocalTerminal
from debug_gym.gym.terminals.terminal import (
    Terminal,
    TerminalError,
    UnrecoverableTerminalError,
)


def test_local_terminal_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("ALLOW_LOCAL_TERMINAL", raising=False)

    with pytest.raises(TerminalError, match="ALLOW_LOCAL_TERMINAL=true"):
        LocalTerminal()


@pytest.mark.parametrize("value", ["", "yes", "1", "tru", "false"])
def test_local_terminal_rejects_disabled_or_malformed_opt_in(monkeypatch, value):
    monkeypatch.setenv("ALLOW_LOCAL_TERMINAL", value)

    with pytest.raises(TerminalError, match="ALLOW_LOCAL_TERMINAL=true"):
        LocalTerminal()


def test_local_terminal_explicit_opt_in_allows_execution(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOW_LOCAL_TERMINAL", "TrUe")

    terminal = LocalTerminal(working_dir=str(tmp_path))

    assert terminal.run("printf enabled") == (True, "enabled")


def test_terminal_run(tmp_path):
    working_dir = str(tmp_path)
    terminal = LocalTerminal(working_dir=working_dir)
    entrypoint = "echo 'Hello World'"
    success, output = terminal.run(entrypoint, timeout=1)
    assert success is True
    assert output == "Hello World"
    assert terminal.working_dir == working_dir


def test_terminal_run_tmp_working_dir():
    terminal = LocalTerminal()
    entrypoint = "pwd -P"
    success, output = terminal.run(entrypoint, timeout=1)
    assert success is True
    assert output == terminal.working_dir


@pytest.mark.parametrize(
    "command",
    [
        ["echo Hello", "echo World"],
        "echo Hello && echo World",
    ],
)
def test_terminal_run_multiple_commands(tmp_path, command):
    working_dir = str(tmp_path)
    terminal = LocalTerminal(working_dir=working_dir)
    success, output = terminal.run(command, timeout=1)
    assert success is True
    assert output == "Hello\nWorld"


def test_terminal_run_failure(tmp_path):
    working_dir = str(tmp_path)
    terminal = LocalTerminal(working_dir=working_dir)
    entrypoint = "ls non_existent_dir"
    success, output = terminal.run(entrypoint, timeout=1)
    assert success is False
    # Linux: "ls: cannot access 'non_existent_dir': No such file or directory"
    # MacOS: "ls: non_existent_dir: No such file or directory"
    pattern = r"ls:.*non_existent_dir.*No such file or directory"
    assert re.search(pattern, output)


def test_terminal_run_timeout(tmp_path):
    """Test that commands that exceed the timeout are killed and return failure."""
    working_dir = str(tmp_path)
    terminal = LocalTerminal(working_dir=working_dir)
    # Run a command that takes longer than the timeout
    entrypoint = "sleep 10 && echo done"
    success, output = terminal.run(entrypoint, timeout=1)
    assert success is False
    assert "timed out" in output.lower()
    assert "1 seconds" in output


def test_terminal_run_default_timeout(tmp_path):
    """Test that the default timeout is applied when none is specified."""
    working_dir = str(tmp_path)
    terminal = LocalTerminal(working_dir=working_dir)
    # Run a quick command without specifying timeout
    entrypoint = "echo 'Hello'"
    success, output = terminal.run(entrypoint)  # No timeout specified
    assert success is True
    assert output == "Hello"
    # Default command_timeout should be 300 seconds (5 minutes)
    assert terminal.command_timeout == 300


def test_terminal_run_custom_command_timeout(tmp_path):
    """Test that custom command_timeout can be set via constructor."""
    working_dir = str(tmp_path)
    terminal = LocalTerminal(working_dir=working_dir, command_timeout=60)
    assert terminal.command_timeout == 60
    # Quick command should still work
    success, output = terminal.run("echo 'test'")
    assert success is True
    assert output == "test"


def test_terminal_output_limit_raises_error(tmp_path):
    """Test that command output exceeding max_output_bytes raises UnrecoverableTerminalError."""
    working_dir = str(tmp_path)
    # Set a small limit to make testing easy
    terminal = LocalTerminal(working_dir=working_dir, max_output_bytes=50)
    # Generate output larger than the limit
    with pytest.raises(UnrecoverableTerminalError, match="exceeded the maximum limit"):
        terminal.run("python3 -c \"print('A' * 200)\"", timeout=5)


def test_terminal_output_no_error_when_under_limit(tmp_path):
    """Test that output under max_output_bytes returns normally."""
    working_dir = str(tmp_path)
    terminal = LocalTerminal(working_dir=working_dir, max_output_bytes=1000)
    success, output = terminal.run("echo 'short output'", timeout=5)
    assert success is True
    assert output == "short output"


def test_terminal_output_limit_disabled(tmp_path):
    """Test that output limit can be disabled with max_output_bytes=0."""
    working_dir = str(tmp_path)
    terminal = LocalTerminal(working_dir=working_dir, max_output_bytes=0)
    success, output = terminal.run("python3 -c \"print('A' * 200)\"", timeout=5)
    assert success is True
    assert len(output) == 200


def test_terminal_output_limit_on_timeout(tmp_path):
    """Test that partial output from timed-out commands also raises error if over limit."""
    working_dir = str(tmp_path)
    terminal = LocalTerminal(working_dir=working_dir, max_output_bytes=50)
    # Command that produces output before sleeping — flush to ensure output is captured
    with pytest.raises(UnrecoverableTerminalError, match="exceeded the maximum limit"):
        terminal.run(
            "python3 -c \"import sys; sys.stdout.write('B' * 200); sys.stdout.flush(); import time; time.sleep(10)\"",
            timeout=2,
        )


def test_terminal_output_limit_error_includes_preview(tmp_path):
    """Test that the error message includes a preview of the output."""
    working_dir = str(tmp_path)
    terminal = LocalTerminal(working_dir=working_dir, max_output_bytes=50)
    with pytest.raises(UnrecoverableTerminalError) as exc_info:
        terminal.run("python3 -c \"print('X' * 200)\"", timeout=5)
    error_msg = str(exc_info.value)
    assert "exceeded the maximum limit" in error_msg
    assert "50 bytes" in error_msg
    assert "Output preview" in error_msg
    assert "XXX" in error_msg  # preview should contain part of the output


def test_terminal_default_max_output_bytes(tmp_path):
    """Test that the default max_output_bytes is set."""
    terminal = LocalTerminal(working_dir=str(tmp_path))
    assert terminal.max_output_bytes == 100_000_000


def test_terminal_session(tmp_path):
    working_dir = str(tmp_path)
    command = "echo Hello World"
    terminal = LocalTerminal(working_dir=working_dir)
    assert not terminal.sessions

    session = terminal.new_shell_session()
    assert len(terminal.sessions) == 1
    output = session.run(command, timeout=1)
    assert output == "Hello World"

    session.run("export TEST_VAR='FooBar'", timeout=1)
    output = session.run("pwd", timeout=1)
    assert output == working_dir
    output = session.run("echo $TEST_VAR", timeout=1)
    assert output == "FooBar"

    terminal.close_shell_session(session)
    assert not terminal.sessions


def test_terminal_multiple_session_commands(tmp_path):
    working_dir = str(tmp_path)
    session_commands = ["echo 'Hello'", "echo 'World'"]
    terminal = LocalTerminal(working_dir, session_commands)
    status, output = terminal.run("pwd", timeout=1)
    assert status
    assert output == f"Hello\nWorld\n{working_dir}"


def test_shell_session_start_with_session_commands(tmp_path):
    terminal = LocalTerminal(
        working_dir=str(tmp_path),
        session_commands=["echo setup"],
    )
    session = terminal.new_shell_session()

    # Test starting without command
    output = session.start()
    assert output == "setup"  # from `echo setup` in session_commands
    assert session.is_running
    assert session.filedescriptor is not None
    assert session.process is not None
    output = session.run("echo Hello World")
    assert output == "Hello World"
    session.close()
    assert not session.is_running
    assert session.filedescriptor is None
    assert session.process is None

    # Test starting with command
    output = session.start("python", ">>>")
    assert output.startswith("setup\r\nPython 3.12")
    assert session.is_running
    assert session.filedescriptor is not None
    assert session.process is not None
    output = session.run("print('test python')", ">>>")
    assert output == "test python"
    session.close()


def test_shell_session_start_without_session_commands(tmp_path):
    terminal = LocalTerminal(working_dir=str(tmp_path))
    session = terminal.new_shell_session()

    # Test starting without command
    output = session.start()
    assert output == ""
    assert session.is_running
    assert session.filedescriptor is not None
    assert session.process is not None
    output = session.run("echo Hello World")
    assert output == "Hello World"
    session.close()
    assert not session.is_running
    assert session.filedescriptor is None
    assert session.process is None

    # Test starting with command
    output = session.start("python", ">>>")
    assert output.startswith("Python 3.12")
    assert session.is_running
    assert session.filedescriptor is not None
    assert session.process is not None
    output = session.run("print('test python')", ">>>")
    assert output == "test python"
    session.close()


def test_copy_content(tmp_path):
    # Create a temporary source file
    source_dir = tmp_path / "source_dir"
    source_dir.mkdir()
    source_file = source_dir / "tmp.txt"
    with open(source_file, "w") as src_file:
        src_file.write("Hello World")

    working_dir = tmp_path / "working_dir"
    working_dir.mkdir()

    terminal = LocalTerminal(working_dir=working_dir)
    # Source must be a folder.
    with pytest.raises(ValueError, match="Source .* must be a directory."):
        terminal.copy_content(source_file)

    terminal.copy_content(source_dir)

    # Clean up the temporary source_dir
    source_file.unlink()
    source_dir.rmdir()

    # Verify the content was copied correctly
    with open(working_dir / "tmp.txt", "r") as f:
        content = f.read()
    assert content == "Hello World"


def test_write_text_preserves_untrusted_content(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOW_LOCAL_TERMINAL", "true")
    terminal = LocalTerminal(working_dir=str(tmp_path))
    side_effect = tmp_path / "side-effect"
    content = (
        "Unicode: café 世界 🚀\n"
        "DEBUGGYM_EOF\n"
        f"$(touch {side_effect}) `touch {side_effect}`; touch {side_effect}\n"
        + ("x" * (2 * 1024 * 1024))
        + "\n"
    )

    terminal.write_text(tmp_path / "payload.txt", content)

    assert (tmp_path / "payload.txt").read_bytes() == content.encode("utf-8")
    assert not side_effect.exists()


def test_base_write_adapter_keeps_content_out_of_shell(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOW_LOCAL_TERMINAL", "true")
    terminal = LocalTerminal(working_dir=str(tmp_path))
    side_effect = tmp_path / "side-effect"
    content = f"DEBUGGYM_EOF\n$(touch {side_effect})\n".encode()

    Terminal.write_bytes(terminal, tmp_path / "adapter.txt", content)

    assert (tmp_path / "adapter.txt").read_bytes() == content
    assert not side_effect.exists()

    executable = tmp_path / "executable"
    executable.write_text("old", encoding="utf-8")
    executable.chmod(0o755)
    Terminal.write_bytes(terminal, executable, b"new")
    assert executable.read_bytes() == b"new"
    assert executable.stat().st_mode & 0o777 == 0o755


def test_write_text_honors_process_umask(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOW_LOCAL_TERMINAL", "true")
    terminal = LocalTerminal(working_dir=str(tmp_path))
    previous_umask = os.umask(0o077)
    try:
        terminal.write_text(tmp_path / "private.txt", "private")
    finally:
        os.umask(previous_umask)

    assert (tmp_path / "private.txt").stat().st_mode & 0o777 == 0o600


def test_write_text_honors_session_umask(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOW_LOCAL_TERMINAL", "true")
    terminal = LocalTerminal(
        working_dir=str(tmp_path),
        session_commands=["umask 0077"],
    )
    previous_umask = os.umask(0o022)
    try:
        terminal.write_text(tmp_path / "private" / "data.txt", "private")
    finally:
        os.umask(previous_umask)

    assert (tmp_path / "private").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "private" / "data.txt").stat().st_mode & 0o777 == 0o600
