# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:43:53 2026

@author: r.aboulhosn
"""

import streamlit as st
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Load-Settlement Model", layout="wide")
st.title("ANN Model for Load–Settlement Response of Drilled Shafts")

# ----------------------------
# INPUTS (REPLACES QLineEdit)
# ----------------------------
D = st.number_input("Pile diameter (mm)", value=457.0)
L = st.number_input("Embedment length (m)", value=15.2)
qct = st.number_input("Cone resistance at tip (MPa)", value=1.9)
frt = st.number_input("Friction ratio tip (%)", value=2.6)
qcs = st.number_input("Cone resistance shaft (MPa)", value=8.6)
frs = st.number_input("Friction ratio shaft (%)", value=0.5)

run = st.button("Run Model")

# ----------------------------
# MODEL
# ----------------------------
def run_model(D, L, qct, frt, qcs, frs):

    Str = [0.0]
    Qi = [0.0]

    inp = [0.0] * 8
    feature2 = [0.0] * 5
    feature4 = [0.0]

    Delta_Ei = 0.1
    Ei = 0.05
    nincr = 29

    inp[0], inp[1], inp[2], inp[3], inp[4], inp[5] = D, L, qct, frt, qcs, frs

    # ---------------- normalization ----------------
    if inp[0] < 305.0: inp[0] = 305.0
    elif inp[0] > 1798.0: inp[0] = 1798.0
    inp[0] = (inp[0] - 305.0) / 1493.0

    if inp[1] < 6.0: inp[1] = 6.0
    elif inp[1] > 27.4: inp[1] = 27.4
    inp[1] = (inp[1] - 6.0) / 21.4

    if inp[2] < 1.9: inp[2] = 1.9
    elif inp[2] > 47.5: inp[2] = 47.5
    inp[2] = (inp[2] - 1.9) / 45.6

    if inp[3] < 0.2: inp[3] = 0.2
    elif inp[3] > 2.63: inp[3] = 2.63
    inp[3] = (inp[3] - 0.2) / 2.43

    if inp[4] < 2.2: inp[4] = 2.2
    elif inp[4] > 28.8: inp[4] = 28.8
    inp[4] = (inp[4] - 2.2) / 26.6

    if inp[5] < 0.46: inp[5] = 0.46
    elif inp[5] > 2.43: inp[5] = 2.43
    inp[5] = (inp[5] - 0.46) / 1.97

    # ---------------- loop ----------------
    for i in range(1, nincr):

        inp[6] = Ei
        inp[7] = Delta_Ei

        inp[6] = min(max(inp[6], 0.05), 20.3)
        inp[6] = (inp[6] - 0.05) / 20.25

        inp[7] = min(max(inp[7], 0.1), 1.45)
        inp[7] = (inp[7] - 0.1) / 1.35

        netsum = -5.424795E-04
        netsum += inp[0] * (-2.392496)
        netsum += inp[1] * (-1.628621)
        netsum += inp[2] * (0.08109)
        netsum += inp[3] * (1.112804)
        netsum += inp[4] * (1.995617E-03)
        netsum += inp[5] * (1.65705)
        netsum += inp[6] * (-1.392372)
        netsum += inp[7] * (2.01394)
        netsum += 0.5048349
        netsum += feature4[0] * (-1.520957)

        feature2[0] = math.tanh(netsum)

        netsum = 4.744489E-03
        netsum = netsum + inp[0] * 1.584351
        netsum = netsum + inp[1] * 0.7739845
        netsum = netsum + inp[2] * (-1.160008)
        netsum = netsum + inp[3] * (-2.688455)
        netsum = netsum + inp[4] * (-0.3640974)
        netsum = netsum + inp[5] * (-0.4922001)
        netsum = netsum + inp[6] * 1.484412;
        netsum = netsum + inp[7] * (-2.994606)
        netsum = netsum + 0.2646977
        netsum = netsum + feature4[0] * 1.188694
        feature2[1]= math.tanh(netsum)

        netsum = -2.335173
        netsum = netsum + inp[0] * 1.627389
        netsum = netsum + inp[1] * (-1.316838)
        netsum = netsum + inp[2] * 1.08041
        netsum = netsum + inp[3] * (-1.604608)
        netsum = netsum + inp[4] * 1.585933
        netsum = netsum + inp[5] * 2.068019
        netsum = netsum + inp[6] * (-0.6480291)
        netsum = netsum + inp[7] * 2.140227
        netsum = netsum + (-1.853716)
        netsum = netsum + feature4[0] * 1.343107
        feature2[2] = math.tanh(netsum)
         
        netsum = 1.580216
        netsum = netsum + inp[0] * 2.050416
        netsum = netsum + inp[1] * (-2.257749)
        netsum = netsum + inp[2] * (-2.876776)
        netsum = netsum + inp[3] * (-1.90816)
        netsum = netsum + inp[4] * (-1.385262)
        netsum = netsum + inp[5] * (-0.2170626)
        netsum = netsum + inp[6] * (-1.655957)
        netsum = netsum + inp[7] * 2.146422
        netsum = netsum + 1.461857
        netsum = netsum + feature4[0] * 1.718647
        feature2[3] = math.tanh(netsum)
         
        netsum = -2.010711
        netsum = netsum + inp[0] * 1.732404
        netsum = netsum + inp[1] * 2.775548
        netsum = netsum + inp[2] * 1.650046
        netsum = netsum + inp[3] * 1.288586
        netsum = netsum + inp[4] * 0.4084347
        netsum = netsum + inp[5] * 0.6821294
        netsum = netsum + inp[6] * (-0.9403843)
        netsum = netsum + inp[7] * 1.237072
        netsum = netsum + (-2.126087)
        netsum = netsum + feature4[0] * (-0.6544736)
        feature2[4] = math.tanh(netsum)
         
        netsum = -0.9554997;
        netsum = netsum + feature2[0] * -1.223897
        netsum = netsum + feature2[1] * -1.75474
        netsum = netsum + feature2[2] * 0.7065521
        netsum = netsum + feature2[3] * 1.694641
        netsum = netsum + feature2[4] * 2.10669
        outp = 1 / (1 + math.exp(-netsum))

        feature4[0] = feature4[0] + feature4[0] * -0.9
        feature4[0] = feature4[0] + outp * 0.9

        outp = 10805.0 *  (outp - 0.1) / 0.8  + 74.0
        
        if (outp<74.0):
            outp = 74.0
        elif (outp>10879.0):
            outp = 10879.0

        Ei += Delta_Ei
        Delta_Ei += 0.05

        if i > 1:
            Str.append(Ei)
            Qi.append(outp)

    return Str, Qi


# ----------------------------
# RUN
# ----------------------------
if run:
    Str, Qi = run_model(D, L, qct, frt, qcs, frs)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(Str, Qi, 'k')
        ax.set_xlabel("Settlement/Diameter (%)")
        ax.set_ylabel("Load (kN)")
        ax.set_xlim([0, 25])
        ax.set_ylim([0, 2000])
        ax.grid(True, linestyle="--")
        st.pyplot(fig)

    with col2:
        df = pd.DataFrame({"Settlement (%)": Str, "Load (kN)": Qi})
        st.dataframe(df)

    st.download_button(
        "Download Excel",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="output.csv",
        mime="text/csv"
    )
    

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: grey;
        font-size: 12px;
        padding: 5px;
        background-color: white;
    }
    </style>

    <div class="footer">
        © 2026 Constructech Services. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
