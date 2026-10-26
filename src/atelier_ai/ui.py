"""Streamlit advisor workflow backed exclusively by the HTTP API."""

from datetime import date
import json
import math
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from atelier_ai.api_models import ReadyResponse
from atelier_ai.ui_client import AdvisorApi, UiApiError, checked_customer_id


METHOD_LABELS = {
    "recent_popularity_7d": "Popularité récente",
    "preferred_product_types": "Types préférés",
    "recent_purchases_180d": "Historique client",
    "popularity_fallback_unknown_customer": "Popularité · client inconnu",
}
MONTHS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre")


def method_label(method):
    return METHOD_LABELS.get(method, "Classement personnalisé")


def french_date(value):
    value = date.fromisoformat(value) if isinstance(value, str) else value
    return f"{value.day} {MONTHS[value.month - 1]} {value.year}"


def number(value):
    return f"{value:,}".replace(",", " ")


@st.cache_data(ttl=30, max_entries=128, show_spinner=False)
def fetch_data(api_url, ready_data, operation, **parameters):
    readiness = ReadyResponse.model_validate(ready_data)
    with AdvisorApi(api_url) as client:
        if operation == "customers":
            return client.customers(readiness, **parameters)
        if operation == "profile":
            return client.profile(readiness, **parameters)
        if operation == "recommendations":
            return client.recommendations(readiness, **parameters)
        if operation == "analysis":
            return client.analysis(readiness)
    raise ValueError("Unknown UI resource")


def _fetch(api_url, ready, operation, **parameters):
    return fetch_data(api_url, ready.model_dump(mode="json"), operation, **parameters)


def _failure(error):
    st.error(str(error))
    if st.button("Réessayer", key="retry_api"):
        fetch_data.clear()
        st.rerun()
    st.stop()


def _customer_picker(api_url, ready):
    with st.sidebar:
        st.subheader("Client")
        mode = st.radio("Rechercher", ["Parcourir", "Identifiant"], horizontal=True, key="customer_mode")
        if mode == "Identifiant":
            with st.form("manual_customer"):
                value = st.text_input("Identifiant client", key="manual_customer_id", max_chars=128)
                submitted = st.form_submit_button("Charger le client", key="load_customer")
            if submitted:
                try:
                    st.session_state["manual_selected"] = checked_customer_id(value)
                    st.session_state.pop("manual_error", None)
                except UiApiError as exc:
                    st.session_state["manual_selected"] = None
                    st.session_state["manual_error"] = str(exc)
            if st.session_state.get("manual_error"):
                st.error(st.session_state["manual_error"])
            return st.session_state.get("manual_selected")
        pages = max(1, math.ceil(ready.customers / 100))
        page_number = st.number_input("Page de clients", min_value=1, max_value=pages, value=1, step=1, key="customer_page")
        page = _fetch(api_url, ready, "customers", offset=(page_number - 1) * 100, limit=100)
        if not page.items:
            st.info("Aucun client sur cette page.")
            return None
        options = [item.customer_id for item in page.items]
        rows = {item.customer_id: item.customer_history_rows for item in page.items}
        def label(customer):
            short = customer if len(customer) <= 20 else f"{customer[:10]}…{customer[-6:]}"
            return f"{short} · {number(rows[customer])} achats observés"
        selected = st.selectbox("Client de la cohorte", options, format_func=label,
                                key=f"customer_browse_{ready.bundle_version}_{page_number}")
        st.caption(f"{number(ready.customers)} clients · page {page_number}/{pages}")
        return selected


def _profile(api_url, ready, customer_id):
    profile = _fetch(api_url, ready, "profile", customer_id=customer_id)
    st.header("Le profil historique")
    if profile is None:
        st.info("Ce client est absent de la cohorte de démonstration. Aucun profil n’est disponible ; la sélection utilisera la popularité.")
        return
    summary = profile.summary
    columns = st.columns(4)
    columns[0].metric("Achats observés · 180 jours", number(summary.customer_history_rows))
    columns[1].metric("Articles distincts", number(summary.customer_distinct_articles))
    columns[2].metric("Dernier achat", "Non observé" if summary.customer_recency_days is None else f"{summary.customer_recency_days:.0f} jours")
    columns[3].metric("Achats · 28 jours", number(summary.customer_rows_28d))
    if not summary.customer_has_history:
        st.info("Aucun achat observé sur les 180 jours précédents. Les recommandations restent disponibles.")
    else:
        st.subheader("Repères")
        left, right = st.columns(2)
        left.write("Groupe le plus observé")
        left.text(summary.dominant_product_group or "Non renseigné")
        right.write(f"{summary.customer_distinct_product_types} types d’articles · {summary.customer_distinct_colours} couleurs observées")
    st.subheader("Derniers achats observés")
    if profile.recent_purchase_rows:
        records = [{"Date": row.t_dat.isoformat(), "Article": row.article_id, "Code canal": row.sales_channel_id}
                   for row in profile.recent_purchase_rows]
        st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
        st.caption("Jusqu’à 20 lignes source, strictement antérieures à la date de référence. Les lignes répétées sont conservées.")
    else:
        st.caption("Aucune ligne d’achat à afficher.")


def _selection(api_url, ready, customer_id):
    st.header("Préparer une sélection")
    preferred = st.session_state.get("preferred_selection_method", ready.default_method)
    default = ready.methods.index(preferred if preferred in ready.methods else ready.default_method)
    method = st.selectbox("Méthode de sélection", ready.methods, index=default,
                          format_func=method_label, key="selection_method")
    st.session_state["preferred_selection_method"] = method
    result = _fetch(api_url, ready, "recommendations", customer_id=customer_id, method=method)
    if result.strategy == "popularity_fallback_unknown_customer":
        st.info("Client absent de la cohorte : voici une sélection de popularité. Aucun profil personnel n’est utilisé.")
    st.caption(f"{method_label(result.strategy)} · 12 propositions · les scores ne sont pas des probabilités d’achat.")
    context = (ready.bundle_version, customer_id, method)
    if st.session_state.get("selection_context") != context:
        st.session_state["selection_context"] = context
        st.session_state["selected_ids"] = [item.article_id for item in result.items]
        for key in list(st.session_state):
            if key.startswith("pick_"):
                del st.session_state[key]
    reset = st.button("Tout retenir", key="select_all")
    if reset:
        st.session_state["selected_ids"] = [item.article_id for item in result.items]
        for key in list(st.session_state):
            if key.startswith("pick_"):
                del st.session_state[key]
    retained = []
    for offset in range(0, len(result.items), 3):
        columns = st.columns(3)
        for column, item in zip(columns, result.items[offset:offset + 3]):
            with column.container(border=True, height="stretch"):
                st.caption(f"PROPOSITION {item.rank:02d}")
                st.text(item.prod_name or item.product_type_name)
                st.caption(f"{item.product_type_name} · {item.colour_group_name}")
                st.text(item.product_group_name)
                st.caption(f"Article {item.article_id}")
                if item.score is not None:
                    score_label = "Achats globaux · 7 jours" if result.score_kind == "historical_purchase_rows_7d" else "Score de classement"
                    st.caption(f"{score_label} : {item.score:.3f}")
                picked = st.checkbox("Retenir", value=item.article_id in st.session_state["selected_ids"],
                                     key=f"pick_{ready.bundle_version}_{customer_id}_{method}_{item.article_id}")
                if picked:
                    retained.append(item)
    st.session_state["selected_ids"] = [item.article_id for item in retained]
    st.subheader(f"{len(retained)} articles retenus")
    export = {
        "customer_id": customer_id, "historical_date": result.historical_date.isoformat(),
        "bundle_version": result.bundle_version, "method_requested": method, "strategy": result.strategy,
        "score_kind": result.score_kind, "source_recommendation_count": 12, "selected_count": len(retained),
        "items": [item.model_dump(mode="json") for item in retained],
    }
    st.download_button("Télécharger la sélection · JSON", json.dumps(export, ensure_ascii=False, indent=2),
                       file_name=f"atelier-selection-{customer_id}.json", mime="application/json",
                       disabled=not retained, key="download_selection")


def _comparison(api_url, ready, customer_id):
    st.header("Comparer les propositions")
    left, right = st.columns(2)
    with left:
        method_a = st.selectbox("Première méthode", ready.methods, format_func=method_label, key="comparison_a")
    with right:
        default = ready.methods.index("recent_purchases_180d") if "recent_purchases_180d" in ready.methods else min(1, len(ready.methods) - 1)
        method_b = st.selectbox("Seconde méthode", ready.methods, index=default, format_func=method_label, key="comparison_b")
    a = _fetch(api_url, ready, "recommendations", customer_id=customer_id, method=method_a)
    b = _fetch(api_url, ready, "recommendations", customer_id=customer_id, method=method_b)
    shared = {item.article_id for item in a.items} & {item.article_id for item in b.items}
    st.metric("Articles en commun", f"{len(shared)} / 12")
    if a.strategy == "popularity_fallback_unknown_customer":
        st.info("Pour ce client inconnu, les deux demandes utilisent la même sélection de popularité.")
    st.caption("Les articles et leurs rangs sont comparés ici. Les échelles de score diffèrent selon la méthode.")
    for column, result in zip(st.columns(2), (a, b)):
        with column:
            st.subheader(method_label(result.strategy))
            rows = [{"Rang": item.rank, "Article": item.article_id, "Nom": item.prod_name or item.product_type_name,
                     "Couleur": item.colour_group_name, "Commun": "Oui" if item.article_id in shared else ""}
                    for item in result.items]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _analysis(api_url, ready):
    result = _fetch(api_url, ready, "analysis")
    st.header("Le bilan V1")
    st.caption(f"Achats du {french_date(result.start_inclusive)} au {french_date(result.end_exclusive)}, fin exclue.")
    metrics = {row.method: row for row in result.comparison}
    model = metrics[ready.default_method]
    baseline = metrics[result.baseline_comparator]
    delta = next(row for row in result.intervals if row.metric == "delta_map_at_12")
    a, b, c, d = st.columns(4)
    a.metric("MAP@12 · modèle", f"{model.map_at_k:.5f}")
    b.metric("MAP@12 · référence", f"{baseline.map_at_k:.5f}")
    c.metric("Écart de MAP@12", f"{delta.observed:+.5f}")
    d.metric("Couverture des candidats", f"{model.candidate_coverage:.2%}")
    if delta.ci_lower <= 0 <= delta.ci_upper:
        st.info("L’intervalle à 95 % recouvre zéro : le gain reste incertain.")
    else:
        st.info("L’intervalle présenté décrit les systèmes ajustés et la semaine observée.")
    st.caption(f"Écart modèle − référence : intervalle à 95 % [{delta.ci_lower:+.5f} ; {delta.ci_upper:+.5f}]. "
               f"Référence choisie sur validation : {method_label(result.baseline_comparator)}.")
    intervals = [row for row in result.intervals if row.metric == "map_at_12"]
    figure = go.Figure()
    for interval in intervals:
        label = method_label(interval.method)
        color = "#176658" if interval.method == ready.default_method else "#B17A30" if interval.method == result.baseline_comparator else "#8C9691"
        figure.add_trace(go.Scatter(x=[interval.ci_lower, interval.ci_upper], y=[label, label],
                                   mode="lines", line={"color": color, "width": 3}, showlegend=False,
                                   hovertemplate="MAP@12 : %{x:.5f}<extra></extra>"))
        figure.add_trace(go.Scatter(x=[interval.observed], y=[label], mode="markers",
                                   marker={"color": color, "size": 11}, showlegend=False,
                                   hovertemplate="MAP@12 observée : %{x:.5f}<extra></extra>"))
    figure.update_layout(title="MAP@12 et intervalles à 95 %", xaxis_title="MAP@12", height=330,
                         margin={"l": 5, "r": 15, "t": 55, "b": 30}, paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", font={"color": "#26362F"})
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    rows = [{"Méthode": method_label(row.method), "MAP@12": f"{row.map_at_k:.5f}",
             "Recall@12": f"{row.recall_at_k:.5f}", "NDCG@12": f"{row.ndcg_at_k:.5f}",
             "Couverture catalogue": f"{row.catalogue_coverage:.2%}",
             "Groupes par sélection": f"{row.mean_recommended_product_groups:.2f}"} for row in result.comparison]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"{number(result.target_buyers)} acheteurs évalués · {number(result.target_pairs)} cibles distinctes · "
               f"{number(result.recipients)} destinataires. {number(result.excluded_nonbuyers)} personnes sans achat cible sont exclues des moyennes de pertinence.")
    st.caption(f"{result.candidate_target_pairs} cibles retrouvées parmi les candidats. Les achats absents des candidats restent dans les dénominateurs. "
               f"{number(result.bootstrap.resamples)} tirages appariés par acheteur, graine {result.bootstrap.seed}.")
    st.caption("V1 a déjà été exposée pendant le développement. La confirmation finale reste réservée ; aucun effet commercial n’est établi.")


def main():
    st.set_page_config(page_title="Atelier AI · Conseil", page_icon="🧵", layout="wide")
    st.caption("ATELIER AI / CONSEIL CLIENT")
    st.title("Préparer une sélection, avec contexte.")
    st.write("Des propositions issues des achats observés, à une date de référence explicite.")
    api_url = os.environ.get("ATELIER_API_URL", "http://127.0.0.1:8000")
    try:
        with AdvisorApi(api_url) as client:
            ready = client.ready()  # Deliberately uncached: never display cached data while readiness is down.
    except UiApiError as exc:
        _failure(exc)
    bundle_context = (api_url, ready.bundle_version)
    if st.session_state.get("bundle_context") != bundle_context:
        for key in list(st.session_state):
            if (key.startswith(("pick_", "customer_browse_"))
                    or key in ("customer_page", "selection_method", "preferred_selection_method", "comparison_a", "comparison_b",
                               "manual_selected", "manual_error", "selection_context", "selected_ids")):
                del st.session_state[key]
        st.session_state["bundle_context"] = bundle_context
    st.caption(f"CATALOGUE HISTORIQUE · {french_date(ready.historical_date).upper()} · "
               "Disponibilité en stock non renseignée")
    view = st.radio("Vue", ["Profil", "Sélection", "Comparaison", "Analyse"], index=1,
                    horizontal=True, key="view")
    with st.sidebar:
        st.title("Atelier AI")
        st.caption("Service connecté")
        if st.button("Actualiser", key="refresh"):
            fetch_data.clear()
            st.rerun()
        with st.expander("Versions"):
            st.caption(f"API {ready.api_version} · bundle {ready.bundle_version[:12]}")
    try:
        customer_id = _customer_picker(api_url, ready)
        if view == "Analyse":
            _analysis(api_url, ready)
        elif customer_id is None:
            st.info("Choisissez un client ou saisissez son identifiant.")
        else:
            with st.expander("Identifiant du client"):
                st.code(customer_id, language=None)
            {"Profil": _profile, "Sélection": _selection, "Comparaison": _comparison}[view](api_url, ready, customer_id)
    except UiApiError as exc:
        _failure(exc)
