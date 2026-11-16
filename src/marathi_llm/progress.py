"""Progress reporting utility ensuring clean line-separated output in logs and terminals."""

import sys
import os
import time
from typing import Optional, Dict, Any
from tqdm import tqdm


class CleanProgressLogger:
    """
    A line-separated progress logger.
    Instead of carriage-return (\\r) updates that collapse together into an
    unreadable continuous line in non-TTY terminals and logs, this prints
    cleanly separated lines at controlled intervals.
    """

    def __init__(
        self,
        total: int,
        desc: str = "Progress",
        unit: str = "item",
        mininterval: float = 3.0,
        file=None,
    ):
        self.total = max(1, total)
        self.desc = desc
        self.unit = unit
        self.mininterval = mininterval
        self.file = file or sys.stdout
        self.n = 0
        self.start_time = time.time()
        self.last_update_time = 0.0
        self.postfix: Dict[str, Any] = {}
        self._print_line()

    def update(self, n: int = 1):
        """Advances the counter by n and prints an update if mininterval elapsed."""
        self.n += n
        now = time.time()
        if (now - self.last_update_time) >= self.mininterval or self.n >= self.total:
            self._print_line()

    def set_postfix(self, postfix_dict: Dict[str, Any], refresh: bool = False):
        """Sets metrics displayed alongside the progress bar."""
        self.postfix = postfix_dict
        if refresh:
            self._print_line()

    def _print_line(self):
        self.last_update_time = time.time()
        elapsed = max(0.001, self.last_update_time - self.start_time)
        rate = self.n / elapsed if self.n > 0 else 0.0
        pct = (self.n / self.total * 100.0) if self.total > 0 else 0.0

        if rate > 0 and self.total > self.n:
            rem_s = (self.total - self.n) / rate
            rem_str = f"{int(rem_s // 60):02d}:{int(rem_s % 60):02d}"
        else:
            rem_str = "00:00"

        el_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        postfix_str = ""
        if self.postfix:
            postfix_str = " | " + ", ".join(f"{k}={v}" for k, v in self.postfix.items())

        rate_str = (
            f"{rate:.1f}{self.unit}/s"
            if rate >= 1.0
            else f"{1.0/rate:.2f}s/{self.unit}"
            if rate > 0
            else f"?{self.unit}/s"
        )

        bar_len = 20
        filled_len = int(bar_len * self.n // self.total)
        bar = "█" * filled_len + " " * (bar_len - filled_len)

        line = (
            f"[{self.desc}] |{bar}| {self.n}/{self.total} ({pct:5.1f}%) "
            f"[{el_str} < {rem_str}, {rate_str}{postfix_str}]"
        )
        self.file.write(line + "\n")
        self.file.flush()

    def close(self):
        """Flushes the final line."""
        if self.n < self.total:
            self._print_line()


def get_progress_bar(
    total: int,
    desc: str = "Progress",
    unit: str = "item",
    mininterval: float = 3.0,
    file=None,
    force_newlines: bool = True,
):
    """
    Factory function returning a progress bar.
    By default (force_newlines=True or when stdout is not a TTY), returns CleanProgressLogger
    which prints clean separate lines with newlines, preventing squashed log output.
    """
    out_file = file or sys.stdout
    is_tty = getattr(out_file, "isatty", lambda: False)()

    # If explicitly requested or running in non-TTY/logs/redirections, use clean lines
    if force_newlines or not is_tty or os.environ.get("PROGRESS_NEWLINES", "1") == "1":
        return CleanProgressLogger(
            total=total,
            desc=desc,
            unit=unit,
            mininterval=mininterval,
            file=out_file,
        )

    # Standard interactive TTY tqdm
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        mininterval=mininterval,
        file=out_file,
        dynamic_ncols=True,
    )
