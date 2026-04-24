import json
import requests
import streamlit as st

GATEWAY_BASE = "http://host.docker.internal:8088"
GRAFANA_URL = "http://localhost:3000"

st.set_page_config(page_title="CAFA5 Demo UI", layout="wide")

st.title("CAFA5 Demo UI")
st.markdown("Kleine Präsentations- und Demo-Oberfläche für den aktuellen CAFA5-Stack.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Links")
    st.markdown(f"- [Grafana]({GRAFANA_URL})")
    st.markdown("- MLflow ist im aktuellen MVP intern angebunden und nicht direkt als Host-Link veröffentlicht.")

with col2:
    st.subheader("Stack")
    st.markdown(
        """
- **Gateway:** NGINX auf Port `8088`
- **Prediction API:** hinter dem Gateway
- **Monitoring:** Prometheus + Grafana separat
- **UI:** Streamlit als Demo-Frontdoor
"""
    )

st.divider()
st.header("Service Status")

def check_url(label: str, url: str):
    try:
        r = requests.get(url, timeout=5)
        ok = r.ok
        st.metric(label, "UP" if ok else f"DOWN ({r.status_code})")
        with st.expander(f"Antwort: {label}"):
            st.code(r.text[:2000] if r.text else f"HTTP {r.status_code}")
    except Exception as e:
        st.metric(label, "DOWN")
        with st.expander(f"Fehler: {label}"):
            st.code(str(e))

status_col1, status_col2 = st.columns(2)

with status_col1:
    check_url("Gateway Health", f"{GATEWAY_BASE}/health")

with status_col2:
    check_url("Prediction API Health", f"{GATEWAY_BASE}/api/predict/health")

st.divider()
st.header("Prediction Demo")

st.markdown(
    "Sende einen Test-Request an den öffentlichen Gateway-Endpunkt `/api/predict`."
)

default_payload = {
    "sequence": "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVLTATQKQNVG"
}

payload_text = st.text_area(
    "JSON Payload",
    value=json.dumps(default_payload, indent=2),
    height=220
)

if st.button("Send Prediction Request"):
    try:
        payload = json.loads(payload_text)
    except Exception as e:
        st.error(f"Ungültiges JSON: {e}")
    else:
        try:
            resp = requests.post(
                f"{GATEWAY_BASE}/api/predict",
                json=payload,
                timeout=60
            )
            st.subheader("Response")
            st.code(resp.text[:6000])
        except Exception as e:
            st.error(f"Request fehlgeschlagen: {e}")

st.divider()
st.header("Demo-Hinweise")
st.markdown(
    """
- Airflow läuft separat auf `8080`
- CAFA5 Gateway läuft auf `8088`
- Grafana läuft auf `3000`
- Streamlit läuft auf `8501`
"""
)
