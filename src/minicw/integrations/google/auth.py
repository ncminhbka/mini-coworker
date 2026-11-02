"""Signing in to Google from a terminal, for Gmail and Calendar.

The reference sends users through a cloud broker (`cloud.py`, `connectors/descriptors.py:541-592`
@ 59151e3, MIT), paused while Google reviews that app. A local user needs their own OAuth client:

  1. Google Cloud console -> a project -> enable "Gmail API" and "Google Calendar API".
  2. OAuth consent screen: "External", status Testing, add your own address as a test user.
  3. Credentials -> create OAuth client ID of type "Desktop app" -> download the JSON.
  4. Save it as `<state>/google/credentials.json`, then run `/google login` in minicw.

`/google login` opens a browser on a free local port and stores the token in
`<state>/google/token.json`. While the consent screen is in Testing, Google expires the refresh
token after 7 days: login again before a demo. The Google libraries are optional
(`pip install "minicw[google]"`) and are imported only here, when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.events",
]
LOGIN_TIMEOUT = 180  # seconds to wait for the browser; this wait cannot be interrupted by Ctrl-C


class GoogleAuthError(Exception):
    pass


def google_dir(state: Path) -> Path:
    return state / "google"


def libraries_installed() -> bool:
    try:
        import google.oauth2.credentials  # noqa: F401
        import google_auth_oauthlib.flow  # noqa: F401
        import googleapiclient.discovery  # noqa: F401
        return True
    except ImportError:
        return False


def status(state: Path) -> str:
    """One line for /google and for `doctor`."""
    if not libraries_installed():
        return 'not available: pip install "minicw[google]"'
    d = google_dir(state)
    if not (d / "credentials.json").is_file():
        return f"no OAuth client yet: save the Desktop-app JSON as {d / 'credentials.json'}"
    if not (d / "token.json").is_file():
        return "ready to sign in: run /google login"
    return "signed in (token saved)"


def login(state: Path) -> str:
    """Run the browser flow and save the token. Blocks until the browser step is finished."""
    if not libraries_installed():
        raise GoogleAuthError('the Google libraries are missing: pip install "minicw[google]"')
    from google_auth_oauthlib.flow import InstalledAppFlow

    d = google_dir(state)
    secrets = d / "credentials.json"
    if not secrets.is_file():
        raise GoogleAuthError(f"missing {secrets} (see the steps in this module's docstring / the README)")
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
    creds = flow.run_local_server(port=0, timeout_seconds=LOGIN_TIMEOUT, open_browser=True)  # port=0: any free port
    d.mkdir(parents=True, exist_ok=True)
    (d / "token.json").write_text(creds.to_json(), encoding="utf-8")
    return "signed in"


def get_credentials(state: Path) -> Any:
    """Valid credentials, refreshed if needed. GoogleAuthError tells the user how to fix it."""
    if not libraries_installed():
        raise GoogleAuthError('the Google libraries are missing: pip install "minicw[google]"')
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token = google_dir(state) / "token.json"
    if not token.is_file():
        raise GoogleAuthError("not signed in to Google: run /google login")
    creds = Credentials.from_authorized_user_file(str(token), SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:  # revoked, or the 7-day limit of a Testing consent screen
            raise GoogleAuthError(f"the Google sign-in expired ({exc}); delete {token} and run /google login") from exc
        token.write_text(creds.to_json(), encoding="utf-8")
        return creds
    raise GoogleAuthError("the Google sign-in is not usable: run /google login")


def build_service(state: Path, api: str, version: str) -> Any:
    from googleapiclient.discovery import build

    return build(api, version, credentials=get_credentials(state), cache_discovery=False)
