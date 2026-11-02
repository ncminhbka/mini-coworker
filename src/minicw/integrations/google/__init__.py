"""Gmail and Google Calendar tools, registered only when they can work."""

from __future__ import annotations

from pathlib import Path

from ...tools.registry import ToolSpec
from .auth import build_service, google_dir, libraries_installed
from .calendar import make_calendar_tools
from .gmail import make_gmail_tools


def make_google_tools(state: Path) -> list[ToolSpec]:
    """The Gmail and Calendar tools, or [] when the Google libraries are not installed or there is
    no OAuth client file yet. Being signed in is checked at call time, so the tools can be offered
    before `/google login` and say what to do when used too early."""
    if not libraries_installed() or not (google_dir(state) / "credentials.json").is_file():
        return []
    return (make_gmail_tools(lambda: build_service(state, "gmail", "v1"))
            + make_calendar_tools(lambda: build_service(state, "calendar", "v3")))
