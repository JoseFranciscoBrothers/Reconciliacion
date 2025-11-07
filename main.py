import pandas as pd
import streamlit as st

SSC_path = "SSCCDATA @1103.XLSX"
diference_path = "110425 (1).XLSX"
inventory_path = "Inventory_loreal_T.XLSX"

def load_SSC_df(path):
    df = pd.read_excel(path,sheet_name="Sheet1")
    df.columns = df.iloc[0]
    df = df.iloc[1:,:]
    df = df.loc[df["Plnt"]=="560"]
    df = df.loc[df["Tk status"]!="41"]
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
    df = pd.read_excel(path,sheet_name="Interface_510")
    df.columns = df.iloc[0]
    df = df.iloc[1:,:13]
    df = df.loc[df["Plant"]=="560"]
    df = df.loc[df["Delta total"].notnull()]
    df = df.loc[df["Delta total"]!=0]
    df = df.loc[df["Storage Location"]=="SL20"]
    df = df.drop(columns=["Plant", "Storage Location", "Base Unit of Measure",
                          "ISIS Available Stock Qty", "WMS/Stock", "Delta",	"Isis not available",
                          "WMS not available","Delta not available"])
    return df

def load_inventory(path):
    df = pd.read_excel(path,sheet_name="Sheet1")
    df = df.loc[df["PLANT2"]=="SL20"]
    df = df.loc[df["HOLD_FLAG"]=="Y"]
    return df


def group_hold(df, holdcode):
    df = df.loc[df["HOLDCODE"] == holdcode]

    grouped = df.groupby('SKU').agg({
        'QTY': "sum",
        'ASN': lambda x: list(x.unique())
    }).reset_index()
    grouped.columns = ["Material_" + holdcode, "Quantity_" + holdcode, "Transport_" + holdcode]

    return grouped


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
comparison_mat = comparison_mat.drop(columns=["Material_DAMAGE", "Material_DIB",
                                              "Material_DIH", "Material_QUA","Material_LOST","Material_STAGE"])

print(comparison_mat)

