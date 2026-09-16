from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_privacy_policy_is_served() -> None:
    """Meta will not publish the app without a reachable privacy-policy
    URL, and an unpublished app receives no Instagram webhooks at all --
    so a 404 here silently costs the deployment its inbound messages.
    Asserts the file is actually found and rendered, not just that a route
    exists: the path is resolved relative to the working directory, which
    is the part that breaks when the Dockerfile stops copying docs/.
    """
    response = client.get("/privacy")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Privacy Policy" in response.text


def test_root_sends_a_visitor_to_the_dashboard() -> None:
    """The bare domain is the link the host's dashboard shows and the one a
    bookmark keeps. Nothing was mounted at "/", so following it answered
    {"detail":"Not Found"} under the platform's own error headers -- which
    reads to a staff member as the clinic's site being down, not as a
    missing route. Asserts the redirect rather than the eventual page: the
    dashboard itself decides whether this visitor gets it or the login form.
    """
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/admin/"
