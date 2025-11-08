import io
import pandas as pd
import streamlit as st
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import ast
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class fillNaColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        values = {
            "Quantity_Status00": 0,
            "Quantity_Status10": 0,
            "Quantity_Status21": 0,
            "Quantity_DAMAGE": 0,
            "Quantity_DIB": 0,
            "Quantity_DIH": 0,
            "Quantity_QUA": 0,
            "Quantity_LOST": 0,
            "Quantity_STAGE": 0,
        }

        list_columns = [
            "Transport_Status00", "Transport_Status10", "Transport_Status21",
            "Transport_DAMAGE", "Transport_DIB", "Transport_DIH",
            "Transport_QUA", "Transport_LOST", "Transport_STAGE"
        ]

        # First fill numeric columns
        X = X.fillna(value=values)

        # Then fill list columns safely with *distinct* empty lists
        for col in list_columns:
            X[col] = X[col].apply(lambda x: [] if isinstance(x, float) and pd.isna(x) else x)

        return X


class addList(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        transport_cols = X.filter(regex=r"^Transport_").columns

        def to_list(x):
            """Safely convert a cell into a list."""
            if isinstance(x, list):
                return x
            if pd.isna(x):
                return []
            if isinstance(x, str):
                s = x.strip()
                if s.startswith('[') and s.endswith(']'):
                    try:
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, list):
                            return parsed
                    except Exception:
                        pass
                # Otherwise split by commas or spaces if looks like multiple items
                if "," in s:
                    return [i.strip() for i in s.split(",") if i.strip()]
                return [s]
            return [x]

        X["Lists"] = X[transport_cols].apply(
            lambda row: sum((to_list(row[col]) for col in transport_cols), []),
            axis=1
        )

        return X


class addManufacturer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):

        def get_manufacturer(values):
            if not values:  # empty list
                return "UNKNOWN"

            has_tr = any(str(v).startswith("TR") for v in values)
            has_non_tr = any(not str(v).startswith("TR") for v in values)

            if has_tr and has_non_tr:
                return "BOTH"
            elif has_tr:
                return "COSBEL"
            elif has_non_tr:
                return "CM"
            else:
                return "UNKNOWN"

        # Apply the function to your combined list column
        X["Manufacturer"] = X["Lists"].apply(get_manufacturer)

        return X


class dropUseless(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        cols_to_drop = list_columns = [
            "Transport_Status00", "Transport_Status10", "Transport_Status21",
            "Transport_DAMAGE", "Transport_DIB", "Transport_DIH",
            "Transport_QUA", "Transport_LOST", "Transport_STAGE", "Lists", "Material"
        ]
        X = X.drop(columns=cols_to_drop)

        return X


class PreprocessColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.preprocessor = None

    def fit(self, X, y=None):
        categorical_features = ['Manufacturer']
        numeric_features = list(set(X.columns) - set(categorical_features))

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Define preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Fit it
        self.preprocessor.fit(X)
        return self

    def transform(self, X, y=None):
        # Apply transformation
        return self.preprocessor.transform(X)


st.set_page_config(page_title="Reconciliación", page_icon="📊", layout="wide")

st.markdown("""
    <style>
        .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

def load_SSC_df(path):
    df = pd.read_excel(path, sheet_name="Sheet1")
    df.columns = df.iloc[0]
    df = df.iloc[1:, :]
    df = df.loc[df["Plnt"] == "560"]
    df = df.loc[df["Tk status"] != "41"]
    return df

def group_status(df, status):
    df = df.loc[df["Tk status"] == status]
    grouped = df.groupby('Material').agg({
        'Packed quantity': "sum",
        'Start': lambda x: list(x.unique())
    }).reset_index()
    grouped.columns = ["Material_" + status, "Quantity_Status" + status, "Transport_Status" + status]
    return grouped

def load_diference(path):
    df = pd.read_excel(path, sheet_name="Interface_510")
    df.columns = df.iloc[0]
    df = df.iloc[1:, :13]
    df = df.loc[df["Plant"] == "560"]
    df = df.loc[df["Delta total"].notnull()]
    df = df.loc[df["Delta total"] != 0]
    df = df.loc[df["Storage Location"] == "SL20"]
    df = df.drop(columns=[
        "Plant", "Storage Location", "Base Unit of Measure",
        "ISIS Available Stock Qty", "WMS/Stock", "Delta", "Isis not available",
        "WMS not available", "Delta not available"
    ])
    return df

def load_inventory(path):
    df = pd.read_excel(path, sheet_name="Sheet1")
    df = df.loc[df["PLANT2"] == "SL20"]
    df = df.loc[df["HOLD_FLAG"] == "Y"]
    return df

def group_hold(df, holdcode):
    df = df.loc[df["HOLDCODE"] == holdcode]
    grouped = df.groupby('SKU').agg({
        'QTY': "sum",
        'ASN': lambda x: list(x.unique())
    }).reset_index()
    grouped.columns = ["Material_" + holdcode, "Quantity_" + holdcode, "Transport_" + holdcode]
    return grouped

def generate(SSC_path, diference_path, inventory_path):
    SSC_df = load_SSC_df(SSC_path)
    grouped_00 = group_status(SSC_df, "00")
    grouped_10 = group_status(SSC_df, "10")
    grouped_21 = group_status(SSC_df, "21")

    diference_df = load_diference(diference_path)

    comparison = pd.merge(diference_df, grouped_00, left_on="Material", right_on="Material_00", how='left')
    comparison = pd.merge(comparison, grouped_10, left_on="Material", right_on="Material_10", how='left')
    comparison = pd.merge(comparison, grouped_21, left_on="Material", right_on="Material_21", how='left')
    comparison = comparison.drop(columns=['Material_00', 'Material_10', 'Material_21'])

    inventory_df = load_inventory(inventory_path)
    grouped_damage = group_hold(inventory_df, "DAMAGE")
    grouped_dib = group_hold(inventory_df, "DIB")
    grouped_dih = group_hold(inventory_df, "DIH")
    grouped_qua = group_hold(inventory_df, "QUA")
    grouped_lost = group_hold(inventory_df, "LOST")
    grouped_stage = group_hold(inventory_df, "STAGE")

    comparison_mat = comparison
    comparison_mat = pd.merge(comparison_mat, grouped_damage, left_on="Material", right_on="Material_DAMAGE", how='left')
    comparison_mat = pd.merge(comparison_mat, grouped_dib, left_on="Material", right_on="Material_DIB", how='left')
    comparison_mat = pd.merge(comparison_mat, grouped_dih, left_on="Material", right_on="Material_DIH", how='left')
    comparison_mat = pd.merge(comparison_mat, grouped_qua, left_on="Material", right_on="Material_QUA", how='left')
    comparison_mat = pd.merge(comparison_mat, grouped_lost, left_on="Material", right_on="Material_LOST", how='left')
    comparison_mat = pd.merge(comparison_mat, grouped_stage, left_on="Material", right_on="Material_STAGE", how='left')
    comparison_mat = comparison_mat.drop(columns=[
        "Material_DAMAGE", "Material_DIB", "Material_DIH", "Material_QUA", "Material_LOST", "Material_STAGE"
    ])

    st.session_state.comparison_mat = comparison_mat



def predict(comparison):

    pipeline_loaded = joblib.load("pipeline_predict.joblib")

    # Predict directly — same as before!
    y_pred = pipeline_loaded.predict(comparison)
    table = comparison["Material"]
    table = comparison[["Material","Delta total"]].copy()  # Ensure it's a DataFrame
    table["Prediccion"] = y_pred
    st.session_state.predict = table

if 'comparison_mat' not in st.session_state:
    st.session_state.comparison_mat = False
    st.session_state.predict = False

with st.sidebar:
    st.header("⚙️ Parámetros")
    SSC_path = st.file_uploader("Archivo de SSC (.xlsx)", type=["xlsx"])
    diference_path = st.file_uploader("Archivo de Diferencias del día (.xlsx)", type=["xlsx"])
    inventory_path = st.file_uploader("Archivo de Inventario (.xlsx)", type=["xlsx"])

    ready = all([SSC_path, diference_path, inventory_path])
    if ready:
        st.success("Archivos listos ✅")
    else:
        st.info("Carga los 3 archivos para continuar")

    if isinstance(st.session_state.get("comparison_mat"), pd.DataFrame):
        if st.button("🗑️ Limpiar resultados", use_container_width=True):
            st.session_state.comparison_mat = False
            st.toast("Resultados limpios", icon="🧹")

st.title("📊 Reconciliación de Inventario")
st.caption("Sube los archivos y genera la tabla de reconciliación.")

with st.form("recon_form", clear_on_submit=False):
    submit = st.form_submit_button("🚀 Generar Reconciliación", disabled=not ready, use_container_width=True)

if submit:
    with st.spinner("Procesando..."):
        try:
            generate(SSC_path, diference_path, inventory_path)
            st.success("Reconciliación generada ✅")
            st.balloons()
        except Exception as e:
            st.error(f"Error: {e}")

if isinstance(st.session_state.get("comparison_mat"), pd.DataFrame) and not st.session_state.comparison_mat.empty:
    st.subheader("📋 Tabla de Reconciliación")
    st.dataframe(st.session_state.comparison_mat, use_container_width=True, height=600)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        st.session_state.comparison_mat.to_excel(writer, index=False, sheet_name='Reconciliacion')
    buffer.seek(0)

    st.download_button(
        "📥 Descargar Excel",
        data=buffer.getvalue(),
        file_name='Reconciliacion.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    with st.form("pred_form", clear_on_submit=False):
        predict_button = st.form_submit_button("🚀 Predecir razón del error", disabled=not ready, use_container_width=True)
    if predict_button:
        with st.spinner("Prediciendo errores..."):
            try:
                predict(st.session_state.comparison_mat)
                st.success("Predicción generada ✅")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")



else:
    st.info("Genera la reconciliación para ver resultados.")


if isinstance(st.session_state.get("predict"), pd.DataFrame) and not st.session_state.predict.empty:
    st.subheader("📋 Predicciónes")
    st.dataframe(st.session_state.predict, use_container_width=True, height=600)
