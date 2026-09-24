"""Purpose: verify experiment commands are shell-safe and dry runs do not execute.
How to run: ``python3 -m pytest tests/test_cluster_commands.py``.
"""

from debour.cluster.commands import format_command, run_command


def test_format_command_quotes_paths() -> None:
    rendered = format_command(["python3", "a file.py", "--name", "hello world"])
    assert rendered == "python3 'a file.py' --name 'hello world'"


def test_dry_run_only_prints(capsys, tmp_path) -> None:
    run_command(["a-command-that-does-not-exist"], execute=False, cwd=tmp_path)
    assert "a-command-that-does-not-exist" in capsys.readouterr().out
