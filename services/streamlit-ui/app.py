import requests
import streamlit as st
import pandas as pd

EMBEDDING_API = "http://host.docker.internal:8000"
PREDICTION_API = "http://host.docker.internal:8001"
GRAFANA_URL = "http://localhost:3000"
NGINX_GATEWAY = "http://host.docker.internal:8088"
NGINX_BROWSER_URL = "http://localhost:8088"

st.set_page_config(page_title="CAFA-5 MLOps Demo", layout="wide")

st.title("CAFA-5 MLOps Demo UI")
st.caption("Protein sequence → ESM2 embedding → GO-term prediction → MLflow champion model")

st.markdown("""
### Stack

- **Embedding API:** `http://localhost:8000`
- **Prediction API:** `http://localhost:8001`
- **MLflow Registry:** `http://localhost:5000`
- **Grafana:** `http://localhost:3000`
- **Prometheus:** `http://localhost:9091`
- **Streamlit UI:** `http://localhost:8501`
""")

st.link_button("Open Grafana", GRAFANA_URL)

def check_url(label, url):
    try:
        r = requests.get(url, timeout=3)
        ok = r.status_code == 200
        st.metric(label, "UP" if ok else "DOWN")
        if not ok:
            st.error(f"{url} returned HTTP {r.status_code}")
        return ok
    except Exception as e:
        st.metric(label, "DOWN")
        st.error(str(e))
        return False

st.subheader("Service Status")
c1, c2 = st.columns(2)
with c1:
    check_url("Embedding API Health", f"{EMBEDDING_API}/api/v1/health")
with c2:
    check_url("Prediction API Health", f"{PREDICTION_API}/health")


st.subheader("Gateway / Security Layer")

st.markdown("""
NGINX is included as a gateway layer for routing and basic authentication.

- **Gateway:** `http://localhost:8088`
- **Public health route:** `/health`
- **Prediction route:** `/api/predict`
- **Protected admin route:** `/admin/mlflow/`
- **Auth concept:** Basic Auth for administrative paths
""")

c_nginx1, c_nginx2 = st.columns(2)

with c_nginx1:
    check_url("NGINX Gateway Health", f"{NGINX_GATEWAY}/health")

with c_nginx2:
    st.link_button("Open NGINX Gateway", NGINX_BROWSER_URL)


st.subheader("Prediction Demo")

sequence = st.text_area(
    "Protein sequence",
    "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVLTATQKQNVG",
    height=120,
)

top_k = st.slider("Top-K GO terms", min_value=3, max_value=20, value=10)

if st.button("Run prediction"):
    payload = {
        "sequences": [
            {
                "id": "demo-sequence",
                "sequence": sequence.strip()
            }
        ],
        "top_k": top_k
    }

    with st.spinner("Running sequence → embedding → GO prediction..."):
        try:
            r = requests.post(
                f"{EMBEDDING_API}/api/v1/predict-go-from-sequences",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    st.success(f"Job status: {data.get('status')} | model version: {data.get('model_version')}")

    results = data.get("results", [])
    if not results:
        st.warning("No results returned.")
        st.json(data)
        st.stop()

    predictions = results[0].get("predictions", [])
    df = pd.DataFrame(predictions)

    st.dataframe(df, use_container_width=True)

    if not df.empty and {"go_term", "score"}.issubset(df.columns):
        st.bar_chart(df.set_index("go_term")["score"])

    with st.expander("Raw response"):
        st.json(data)

st.markdown("""
### Demo Notes

This UI is intentionally thin: it does not load models itself.  
It only calls the deployed services and visualizes their responses.
""")
