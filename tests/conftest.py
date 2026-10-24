"""Global pytest configuration and fixtures."""
import sys
import os
import pytest

# Ensure Tkinter on Windows Python 3.13 always finds standard library files
base = sys.base_prefix
tcl_dir = os.path.join(base, "tcl", "tcl8.6")
tk_dir = os.path.join(base, "tcl", "tk8.6")
if os.path.exists(tcl_dir):
    os.environ["TCL_LIBRARY"] = tcl_dir
if os.path.exists(tk_dir):
    os.environ["TK_LIBRARY"] = tk_dir


@pytest.fixture(scope="session")
def session_tk_root():
    """Provides a shared hidden Tk root for the entire test session."""
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass
