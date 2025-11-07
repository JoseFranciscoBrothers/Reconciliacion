import io
import pandas as pd
import streamlit as st

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

if 'comparison_mat' not in st.session_state:
    st.session_state.comparison_mat = False

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
else:
    st.info("Genera la reconciliación para ver resultados.")
