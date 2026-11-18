import io
import pytest
from marathi_llm.progress import CleanProgressLogger, get_progress_bar


def test_clean_progress_logger_separated_lines():
    buf = io.StringIO()
    pbar = CleanProgressLogger(total=10, desc="Test Task", mininterval=0.0, file=buf)
    for i in range(10):
        pbar.set_postfix({"loss": f"{1.0/(i+1):.2f}"})
        pbar.update(1)
    pbar.close()

    output = buf.getvalue()
    lines = [line for line in output.split("\n") if line.strip()]

    # Verify that multiple lines were printed and every update is on its own separate line
    assert len(lines) > 5
    for line in lines:
        assert line.startswith("[Test Task]")
        assert "%" in line
        assert "|" in line

    # Final line should indicate 100% completion
    assert "10/10" in lines[-1]
    assert "100.0%" in lines[-1]


def test_get_progress_bar_factory():
    buf = io.StringIO()
    pbar = get_progress_bar(total=5, desc="Factory Test", file=buf, force_newlines=True)
    assert isinstance(pbar, CleanProgressLogger)
    pbar.update(5)
    pbar.close()
    assert "Factory Test" in buf.getvalue()


def test_clean_progress_logger_throttles_past_total():
    buf = io.StringIO()
    pbar = CleanProgressLogger(total=5, desc="Overflow", mininterval=60.0, file=buf)
    for _ in range(20):
        pbar.update(1)
    pbar.close()

    lines = [line for line in buf.getvalue().split("\n") if line.strip()]
    # Only initial line (0/5) and exact total (5/5) should print; updates 6..20 are throttled
    assert len(lines) == 2
    assert "5/5" in lines[-1]
    assert "|████████████████████|" in lines[-1]

