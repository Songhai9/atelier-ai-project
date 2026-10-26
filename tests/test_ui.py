"""Exercise the advisor workflow against the real API contract."""

from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from atelier_ai.api import create_app
import atelier_ai.ui as ui
from atelier_ai.ui_client import AdvisorApi, UiApiError
from test_api import api_bundle, evaluation_campaign, serving_campaign


@pytest.fixture
def browser_app(api_bundle, monkeypatch):
    _, directory = api_bundle
    ui.fetch_data.clear()
    with TestClient(create_app(directory)) as server:
        def respond(request):
            return server.request(request.method, str(request.url.copy_with(scheme="http", host="testserver")))

        def client(_):
            return AdvisorApi("http://testserver", transport=httpx.MockTransport(respond))

        monkeypatch.setattr(ui, "AdvisorApi", client)
        monkeypatch.setenv("ATELIER_API_URL", "http://testserver")
        yield AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"),
                                default_timeout=20).run()
    ui.fetch_data.clear()


def test_selection_survives_navigation_and_resets_for_another_customer(browser_app):
    app = browser_app
    assert not app.exception
    assert len(app.checkbox) == 12 and all(box.value for box in app.checkbox)
    assert any("5 AOÛT 2020" in item.value for item in app.caption)
    app.checkbox[0].uncheck().run()
    assert len(app.session_state["selected_ids"]) == 11
    app.radio(key="view").set_value("Comparaison").run()
    assert not app.exception and len(app.dataframe) == 2
    app.radio(key="view").set_value("Sélection").run()
    assert not app.exception and sum(box.value for box in app.checkbox) == 11
    app.sidebar.selectbox[0].select_index(2).run()
    assert not app.exception and sum(box.value for box in app.checkbox) == 12
    app.selectbox(key="selection_method").select_index(3).run()
    preferred = app.selectbox(key="selection_method").value
    app.checkbox[0].uncheck().run()
    app.radio(key="view").set_value("Profil").run()
    assert not app.exception
    app.radio(key="view").set_value("Sélection").run()
    assert not app.exception
    assert app.selectbox(key="selection_method").value == preferred
    assert sum(box.value for box in app.checkbox) == 11
    app.sidebar.selectbox[0].select_index(0).run()
    assert sum(box.value for box in app.checkbox) == 12


def test_unknown_and_invalid_manual_customer_have_explicit_states(browser_app):
    app = browser_app
    app.radio(key="customer_mode").set_value("Identifiant").run()
    app.text_input(key="manual_customer_id").set_value("outside-cohort")
    app.button(key="load_customer").click().run()
    assert not app.exception and len(app.checkbox) == 12
    assert any("Client absent de la cohorte" in item.value for item in app.info)
    app.radio(key="view").set_value("Profil").run()
    assert not app.exception and not app.metric
    assert any("absent de la cohorte" in item.value for item in app.info)
    app.text_input(key="manual_customer_id").set_value("invalid/id")
    app.button(key="load_customer").click().run()
    assert not app.exception and app.error
    assert app.session_state["manual_selected"] is None and not app.checkbox


def test_cold_profile_and_analysis_show_historical_limits(browser_app):
    app = browser_app
    app.radio(key="view").set_value("Profil").run()
    assert not app.exception
    assert any("Aucun achat observé" in item.value for item in app.info)
    app.radio(key="view").set_value("Analyse").run()
    assert not app.exception and len(app.get("plotly_chart")) == 1
    assert any("gain reste incertain" in item.value for item in app.info)
    assert any("2 acheteurs évalués · 3 cibles distinctes" in item.value for item in app.caption)
    assert any("confirmation finale reste réservée" in item.value for item in app.caption)


def test_api_failure_hides_previous_selection(browser_app, monkeypatch):
    app = browser_app
    assert len(app.checkbox) == 12

    class Offline:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def ready(self):
            raise UiApiError("Le service est indisponible.", code="unreachable")

    monkeypatch.setattr(ui, "AdvisorApi", lambda _: Offline())
    app.run()
    assert not app.exception and app.error
    assert not app.checkbox
    assert app.button(key="retry_api")


def test_invalid_api_configuration_is_actionable(monkeypatch):
    monkeypatch.setenv("ATELIER_API_URL", "file:///invalid")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py")).run()
    assert not app.exception and app.error
    assert "adresse" in app.error[0].value
