"""Helpers to authenticate against the AWBW website using requests + BeautifulSoup."""

from __future__ import annotations

from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict

BASE_URL = "https://awbw.amarriner.com/"
LOGIN_CHECK_PATH = "logincheck.php"


class AWBWLoginError(RuntimeError):
    """Raised when the login attempt is rejected by the AWBW backend."""


class AWBWAuthResult(BaseModel):
    """Encapsulates the outcome of an authentication attempt."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: requests.Session
    message: Optional[str] = None


def _validate_login_form(html: str) -> None:
    """Ensure the expected login form is still present on the page.

    Parameters
    ----------
    html:
        Raw HTML from the AWBW landing page.

    Raises
    ------
    AWBWLoginError
        If the login form cannot be located (structure may have changed).
    """

    soup = BeautifulSoup(html, "html.parser")
    form = soup.select_one("form.login-form")
    if not form:
        raise AWBWLoginError(
            "Could not locate the AWBW login form structure. The page layout may "
            "have changed, or access is blocked."
        )


def authenticate(username: str, password: str) -> AWBWAuthResult:
    """Authenticate a user and return an HTTP session populated with AWBW cookies.

    This mirrors the browser flow: we pull the landing page to confirm the login
    form exists, then POST credentials to `logincheck.php`. A response body of
    ``\"1\"`` indicates success; any other payload is surfaced as an error.
    """

    if not username or not password:
        raise ValueError("username and password must both be provided")

    session = requests.Session()
    landing = session.get(BASE_URL, timeout=15)
    landing.raise_for_status()
    _validate_login_form(landing.text)

    login_url = urljoin(BASE_URL, LOGIN_CHECK_PATH)
    response = session.post(
        login_url,
        data={"username": username, "password": password},
        timeout=15,
        headers={"Referer": BASE_URL},
    )
    response.raise_for_status()

    payload = response.text.strip()
    if payload != "1":
        raise AWBWLoginError(payload or "Unknown login failure")

    return AWBWAuthResult(session=session, message="Login successful")
