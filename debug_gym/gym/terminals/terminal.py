import atexit
import shlex
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from debug_gym.gym.terminals.shell_session import DEFAULT_PS1
from debug_gym.logger import DebugGymLogger


class TerminalError(RuntimeError):
    """Base exception for terminal-related failures."""


class UnrecoverableTerminalError(TerminalError):
    """Raised when the terminal becomes unusable and the episode must stop."""

    def __init__(self, message: str, env_info=None):
        super().__init__(message)
        self.env_info = env_info


DISABLE_ECHO_COMMAND = "stty -echo"

# Default cap on command output to prevent unbounded memory/disk usage.
# Commands producing more output than this will have their output truncated.
DEFAULT_MAX_OUTPUT_BYTES = 100_000_000  # 100 MB


class Terminal(ABC):

    def __init__(
        self,
        working_dir: str | None = None,
        session_commands: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        logger: DebugGymLogger | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        **kwargs,
    ):
        self.logger = logger or DebugGymLogger("debug-gym")
        self.session_commands = session_commands or []
        self.env_vars = env_vars or {}
        # Clean up output by disabling terminal prompt and colors
        self.env_vars["NO_COLOR"] = "1"  # disable colors
        self.env_vars["PYTHONSTARTUP"] = ""  # prevent Python from loading startup files
        # use a sentinel to know when to stop reading
        self.env_vars["PS1"] = DEFAULT_PS1
        self.env_vars["PYTHONDONTWRITEBYTECODE"] = "1"  # prevent creation of .pyc files

        self._working_dir = working_dir
        self.max_output_bytes = max_output_bytes
        self.sessions = []

        kwargs.pop("type", None)  # remove 'type' if present
        if kwargs:
            self.logger.debug(f"Ignoring unknown parameters: {kwargs}")

    @property
    def working_dir(self):
        """Lazy initialization of the working directory."""
        if self._working_dir is None:
            _tempdir = tempfile.TemporaryDirectory(prefix="Terminal-")
            atexit.register(_tempdir.cleanup)
            self._working_dir = str(Path(_tempdir.name).resolve())
            self.logger.debug(f"Using temporary working directory: {self._working_dir}")
        return self._working_dir

    @working_dir.setter
    def working_dir(self, value):
        self._working_dir = value

    @abstractmethod
    def prepare_command(self, entrypoint: str | list[str]) -> list[str]:
        """Prepares a shell command by combining session commands and entrypoint commands.
        Then wraps the command in a shell (self.default_shell_command) call."""
        pass

    @abstractmethod
    def run(
        self,
        entrypoint: str | list[str],
        timeout: int = None,
        raises: bool = False,
        strip_output: bool = True,
    ) -> tuple[bool, str]:
        """Run a list of commands in the terminal. Return command status and output."""
        pass

    @property
    @abstractmethod
    def default_shell_command(self) -> str:
        pass

    @abstractmethod
    def new_shell_session(self):
        pass

    def close_shell_session(self, session):
        session.close()
        self.sessions.remove(session)

    def close(self):
        for session in self.sessions:
            self.close_shell_session(session)

    def _truncate_output(self, output: str) -> str:
        """Truncate command output to max_output_bytes to prevent unbounded memory/disk usage."""
        if self.max_output_bytes > 0 and len(output) > self.max_output_bytes:
            original_len = len(output)
            output = (
                output[: self.max_output_bytes]
                + f"\n\n[OUTPUT TRUNCATED: {original_len} bytes -> {self.max_output_bytes} bytes]"
            )
        return output

    def _raise_output_limit_exceeded(self, total_bytes: int, preview: str = "") -> None:
        """Raise UnrecoverableTerminalError when command output exceeds the limit."""
        truncated_preview = preview[:2000] if preview else ""
        msg = (
            f"Command output exceeded the maximum limit of "
            f"{self.max_output_bytes} bytes (got at least {total_bytes} bytes). "
            f"Terminating to prevent resource exhaustion."
        )
        if truncated_preview:
            msg += f"\nOutput preview (first 2000 chars):\n{truncated_preview}"
        raise UnrecoverableTerminalError(msg)

    def __str__(self):
        return f"Terminal[{self.working_dir}]"

    def write_text(
        self, filepath: str | Path, content: str, encoding: str = "utf-8"
    ) -> None:
        """Write text without embedding it in a shell command."""
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        self.write_bytes(filepath, content.encode(encoding))

    def _compatibility_write_metadata(
        self, target: Path
    ) -> tuple[int, int, int, int, int]:
        user_success, user_output = self.run("id -u", raises=False)
        group_success, group_output = self.run("id -g", raises=False)
        groups_success, groups_output = self.run("id -G", raises=False)
        if not user_success or not group_success or not groups_success:
            raise TerminalError("Failed to determine the terminal user")
        runtime_user_id = int(user_output.splitlines()[-1])
        runtime_group_id = int(group_output.splitlines()[-1])
        runtime_group_ids = {
            int(group) for group in groups_output.splitlines()[-1].split()
        }

        exists, _ = self.run(
            f"test -e {shlex.quote(str(target))}",
            raises=False,
        )
        if exists:
            is_regular, _ = self.run(
                f"test -f {shlex.quote(str(target))}",
                raises=False,
            )
            if not is_regular:
                raise TerminalError("Write target must be a regular file")
            metadata_success, metadata_output = self.run(
                f"stat -c '%a %u %g' -- {shlex.quote(str(target))}",
                raises=False,
            )
            if not metadata_success:
                raise TerminalError("Failed to inspect destination file")
            mode, user_id, group_id = metadata_output.splitlines()[-1].split()
            user_id = int(user_id)
            group_id = int(group_id)
            if runtime_user_id != 0 and (
                user_id != runtime_user_id
                or (
                    group_id not in runtime_group_ids
                    and not self._compatibility_inherits_parent_group(
                        target.parent, group_id
                    )
                )
            ):
                raise TerminalError("Cannot preserve destination file ownership")
            return (
                int(mode, 8) & ~0o6000,
                user_id,
                group_id,
                runtime_user_id,
                runtime_group_id,
            )

        umask_success, umask_output = self.run("umask", raises=False)
        parent_success, parent_output = self.run(
            f"stat -c '%a %g' -- {shlex.quote(str(target.parent))}",
            raises=False,
        )
        if not umask_success or not parent_success:
            raise TerminalError("Failed to determine destination file metadata")
        parent_mode, parent_group_id = parent_output.splitlines()[-1].split()
        group_id = (
            int(parent_group_id) if int(parent_mode, 8) & 0o2000 else runtime_group_id
        )
        mode = 0o666 & ~int(umask_output.splitlines()[-1], 8)
        return (
            mode,
            runtime_user_id,
            group_id,
            runtime_user_id,
            runtime_group_id,
        )

    def _compatibility_inherits_parent_group(self, parent: Path, group_id: int) -> bool:
        success, output = self.run(
            f"stat -c '%a %g' -- {shlex.quote(str(parent))}",
            raises=False,
        )
        if not success:
            raise TerminalError("Failed to inspect destination directory")
        parent_mode, parent_group_id = output.splitlines()[-1].split()
        return bool(int(parent_mode, 8) & 0o2000 and int(parent_group_id) == group_id)

    def write_bytes(self, filepath: str | Path, content: bytes) -> None:
        """Compatibility transport for backends that implement copy_content."""
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")

        target = Path(filepath)
        temporary_name = f".debug-gym-write-{uuid.uuid4().hex}.tmp"
        remote_temporary_path = target.parent / temporary_name
        with tempfile.TemporaryDirectory(prefix="DebugGym-write-") as staging:
            staging_path = Path(staging)
            (staging_path / temporary_name).write_bytes(content)

            success, output = self.run(
                f"mkdir -p -- {shlex.quote(str(target.parent))}",
                raises=False,
            )
            if not success:
                raise TerminalError(f"Failed to create destination directory: {output}")

            (
                mode,
                user_id,
                group_id,
                runtime_user_id,
                runtime_group_id,
            ) = self._compatibility_write_metadata(target)
            try:
                self.copy_content(staging_path, target.parent)
                metadata_commands = [
                    f"chmod {mode:o} -- " f"{shlex.quote(str(remote_temporary_path))}"
                ]
                if runtime_user_id == 0:
                    metadata_commands.append(
                        f"chown {user_id}:{group_id} -- "
                        f"{shlex.quote(str(remote_temporary_path))}"
                    )
                elif group_id != runtime_group_id:
                    metadata_commands.append(
                        "(test "
                        f'"$(stat -c %g -- '
                        f'{shlex.quote(str(remote_temporary_path))})" '
                        f"= {group_id} || chgrp {group_id} -- "
                        f"{shlex.quote(str(remote_temporary_path))})"
                    )
                metadata_commands.append(
                    "mv -f -- "
                    f"{shlex.quote(str(remote_temporary_path))} "
                    f"{shlex.quote(str(target))}"
                )
                success, output = self.run(
                    " && ".join(metadata_commands),
                    raises=False,
                )
                if not success:
                    raise TerminalError(f"Failed to replace destination file: {output}")
            finally:
                self.run(
                    f"rm -f -- {shlex.quote(str(remote_temporary_path))}",
                    raises=False,
                )

    @abstractmethod
    def copy_content(self, src: str | Path, target: str | Path | None = None) -> None:
        """Copy files contained in src on the host to target on the host."""
        pass
