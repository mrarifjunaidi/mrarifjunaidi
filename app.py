"""
Aplikasi Prediksi Harga Rumah - Gradient Boosting (Orange Data Mining)
Model: GradientBoostingRegressor via Orange3
Deployment: Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Path model — gunakan path relatif agar kompatibel dengan Streamlit Cloud
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "Model_Gradient_Boost_Harga_Rumah.pkcls"

# ---------------------------------------------------------------------------
# Konfigurasi fitur
# Nama fitur HARUS sama persis dengan nama variabel saat training di Orange
# ---------------------------------------------------------------------------
FEATURE_CONFIG = {
    "X1 transaction date": {
        "label": "Tanggal Transaksi (tahun desimal, mis. 2013.5)",
        "type": "numeric",
        "input": "number",
        "min": 2012.0,
        "max": 2014.0,
        "default": 2013.5,
        "step": 0.083,          # ~1 bulan
        "help": "Format tahun desimal: 2013.0 = Jan 2013, 2013.5 = Jul 2013",
    },
    "X2 house age": {
        "label": "Usia Rumah (tahun)",
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 50,
        "default": 10,
        "step": 1,
        "help": "Usia bangunan dalam tahun",
    },
    "X3 distance to the nearest MRT station": {
        "label": "Jarak ke Stasiun MRT Terdekat (meter)",
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 7000,
        "default": 500,
        "step": 10,
        "help": "Jarak dalam meter ke stasiun MRT terdekat",
    },
    "X4 number of convenience stores": {
        "label": "Jumlah Convenience Store di Sekitar",
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 15,
        "default": 5,
        "step": 1,
        "help": "Jumlah minimarket/convenience store dalam radius tertentu",
    },
    "X5 latitude": {
        "label": "Latitude Lokasi",
        "type": "numeric",
        "input": "number",
        "min": 24.9,
        "max": 25.1,
        "default": 25.0,
        "step": 0.0001,
        "help": "Koordinat latitude lokasi rumah (sekitar wilayah Taiwan)",
    },
}

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model machine learning...")
def load_model():
    """Load model Orange dari file .pkcls menggunakan pickle."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}\n"
            "Pastikan file 'Model_Gradient_Boost_Harga_Rumah.pkcls' "
            "berada di direktori yang sama dengan app.py di repository GitHub."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


# ---------------------------------------------------------------------------
# Fungsi prediksi via Orange
# ---------------------------------------------------------------------------
def predict_with_orange(model, input_values: list) -> float:
    """
    Jalankan prediksi menggunakan Orange domain dan Table.

    Parameters
    ----------
    model       : model Orange (SklModelRegression)
    input_values: list nilai numerik sesuai urutan FEATURE_CONFIG

    Returns
    -------
    float: nilai prediksi harga rumah per unit area
    """
    try:
        import Orange.data  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Library 'orange3' belum terinstall. "
            "Pastikan orange3 ada di requirements.txt."
        ) from exc

    domain = model.domain

    # Buat domain baru dengan semua fitur sebagai ContinuousVariable
    # (X1 adalah TimeVariable di Orange, tapi nilainya numerik)
    attrs = [Orange.data.ContinuousVariable(a.name) for a in domain.attributes]
    class_var = Orange.data.ContinuousVariable(domain.class_var.name)
    new_domain = Orange.data.Domain(attrs, class_var)

    X = np.array([input_values], dtype=float)
    Y = np.zeros((1, 1), dtype=float)   # placeholder, tidak dipakai

    table = Orange.data.Table.from_numpy(new_domain, X, Y)
    predictions = model(table)
    return float(predictions[0])


def predict_with_sklearn_fallback(model, input_df: pd.DataFrame) -> float:
    """
    Fallback: coba langsung model.predict() ala scikit-learn.
    Berguna jika model meng-expose skl_model langsung.
    """
    if hasattr(model, "skl_model"):
        return float(model.skl_model.predict(input_df.values)[0])
    return float(model.predict(input_df.values)[0])


# ---------------------------------------------------------------------------
# UI Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("ℹ️ Panduan Penggunaan")
    st.markdown(
        """
        1. **Isi semua parameter** properti pada form di halaman utama.
        2. Klik tombol **🔮 Prediksi Harga** untuk mendapatkan hasil.
        3. Hasil ditampilkan dalam satuan **harga per unit area**.

        ---
        **Tentang Model**
        - Algoritma: *Gradient Boosting Regressor*
        - Training: Orange Data Mining
        - Model disimpan sebagai file `.pkcls` di repository GitHub yang sama.

        ---
        **Fitur Model**
        | Kode | Deskripsi |
        |------|-----------|
        | X1 | Tanggal transaksi |
        | X2 | Usia rumah |
        | X3 | Jarak ke MRT |
        | X4 | Jumlah convenience store |
        | X5 | Latitude lokasi |

        **Target (Y):** Harga rumah per unit area
        """
    )

# ---------------------------------------------------------------------------
# Halaman Utama
# ---------------------------------------------------------------------------
st.title("🏠 Prediksi Harga Rumah")
st.markdown(
    """
    Aplikasi ini menggunakan model **Gradient Boosting** hasil training dari
    **Orange Data Mining** dan dijalankan melalui **Streamlit Cloud**.

    Masukkan informasi properti di bawah ini, lalu klik tombol prediksi.
    """
)
st.divider()

# ---------------------------------------------------------------------------
# Form Input
# ---------------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("📋 Data Properti")

    input_data = {}

    for feature_name, cfg in FEATURE_CONFIG.items():
        label = cfg["label"]
        help_text = cfg.get("help", "")

        if cfg["type"] == "numeric":
            if cfg["input"] == "slider":
                input_data[feature_name] = st.slider(
                    label=label,
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(cfg["default"]),
                    step=float(cfg.get("step", 1)),
                    help=help_text,
                )
            elif cfg["input"] == "number":
                input_data[feature_name] = st.number_input(
                    label=label,
                    min_value=float(cfg["min"]),
                    max_value=float(cfg["max"]),
                    value=float(cfg["default"]),
                    step=float(cfg.get("step", 1)),
                    help=help_text,
                    format="%.4f" if cfg.get("step", 1) < 1 else "%.2f",
                )
        elif cfg["type"] == "categorical":
            input_data[feature_name] = st.selectbox(
                label=label,
                options=cfg["options"],
                help=help_text,
            )

    submitted = st.form_submit_button("🔮 Prediksi Harga", use_container_width=True)

# ---------------------------------------------------------------------------
# Proses Prediksi
# ---------------------------------------------------------------------------
if submitted:
    # Tampilkan ringkasan input
    st.subheader("📊 Data yang Diinputkan")
    display_df = pd.DataFrame(
        {
            "Fitur": list(FEATURE_CONFIG.keys()),
            "Deskripsi": [cfg["label"] for cfg in FEATURE_CONFIG.values()],
            "Nilai": [input_data[k] for k in FEATURE_CONFIG.keys()],
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Load model
    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(f"❌ **Model tidak ditemukan!**\n\n{e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Gagal memuat model!**\n\n{e}")
        st.stop()

    # Jalankan prediksi
    input_values = [float(input_data[k]) for k in FEATURE_CONFIG.keys()]

    with st.spinner("Menghitung prediksi..."):
        try:
            # Pendekatan utama: Orange domain
            result = predict_with_orange(model, input_values)
            method_used = "Orange (primary)"

        except Exception as orange_err:
            st.warning(f"⚠️ Orange prediction gagal: `{orange_err}` — mencoba fallback scikit-learn...")
            try:
                input_df = pd.DataFrame(
                    [input_data],
                    columns=list(FEATURE_CONFIG.keys()),
                )
                result = predict_with_sklearn_fallback(model, input_df)
                method_used = "scikit-learn fallback"
            except Exception as skl_err:
                st.error(
                    f"❌ **Prediksi gagal!**\n\n"
                    f"- Orange error: `{orange_err}`\n"
                    f"- Sklearn error: `{skl_err}`\n\n"
                    "Periksa apakah nama kolom input sesuai dengan fitur saat training "
                    "atau hubungi administrator."
                )
                st.stop()

    # Tampilkan hasil
    st.divider()
    st.subheader("🎯 Hasil Prediksi")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.success(
            f"**Prediksi Harga Rumah per Unit Area**\n\n"
            f"### {result:,.2f}"
        )
    with col2:
        st.metric(
            label="Nilai Prediksi",
            value=f"{result:,.2f}",
            help="Harga rumah per unit area berdasarkan model Gradient Boosting",
        )

    st.caption(f"_Metode prediksi yang digunakan: {method_used}_")

    st.info(
        "💡 **Catatan:** Nilai prediksi merupakan estimasi berdasarkan model machine learning. "
        "Hasil aktual dapat berbeda tergantung kondisi pasar dan faktor lain yang tidak termasuk dalam model."
    )
