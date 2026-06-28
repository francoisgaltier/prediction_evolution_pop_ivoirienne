"""
Simulateur PND Démographique — Côte d'Ivoire
Intégration du module d'ingestion ONP + calibration automatique
"""
import streamlit as tk
import numpy as np
import pandas as pd
import plotly.express as px

# Import du module ONP (doit être dans le même répertoire)
from onp_ingestion import (
    charger_donnees_integrees, executer_modele, calculer_matrice_distances,
    afficher_page_ingestion
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
tk.set_page_config(page_title="Simulateur PND Démographique", layout="wide")

# Navigation multi-pages
page = tk.sidebar.radio(
    "Navigation",
    ["🗺️ Simulateur Principal", "📥 Ingestion & Calibration ONP"],
    label_visibility="collapsed"
)

# ─────────────────────────────────────────────────────────────
# PAGE 2 — MODULE ONP (délégué au module dédié)
# ─────────────────────────────────────────────────────────────
if page == "📥 Ingestion & Calibration ONP":
    afficher_page_ingestion()
    tk.stop()

# ─────────────────────────────────────────────────────────────
# PAGE 1 — SIMULATEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────
tk.title("🇨🇮 Simulateur d'Aide à la Décision PND — Modèle Gravitaire à Somme Nulle")
tk.markdown("Calibré sur l'historique officiel **RGPH 2014** · Validation croisée **RGPH 2021**.")

# Chargement des données (issues de l'ingestion ONP si disponibles, sinon RGPH intégré)
df = tk.session_state.get("df_onp", charger_donnees_integrees())
NOMS   = df["nom"].tolist()
LATS   = df["latitude"].values
LONS   = df["longitude"].values
POP_INIT = df["pop_recensement_1"].values.astype(float)
POP_REEL = df["pop_recensement_2"].values.astype(float)
N_POLES  = len(df)
A1 = int(df["annee_recensement_1"].iloc[0]) if "annee_recensement_1" in df.columns else 2014
A2 = int(df["annee_recensement_2"].iloc[0]) if "annee_recensement_2" in df.columns else 2021

MATRICE_DISTANCES = calculer_matrice_distances(LATS, LONS)

# Paramètres calibrés (issus de l'onglet calibration, ou valeurs par défaut)
PARAMS_DEFAULT = dict(alpha=1/15, beta=1/50, gamma=0.075,
                      expo_dist=1.5, taux_emig=0.03, prime_cap=0.015)
params_actifs = tk.session_state.get("params_calibres", PARAMS_DEFAULT)

if "params_calibres" in tk.session_state:
    tk.sidebar.success("✅ Paramètres calibrés ONP actifs")
else:
    tk.sidebar.info("ℹ️ Paramètres par défaut — Allez dans **Ingestion & Calibration ONP** pour calibrer")

# ─────────────────────────────────────────────────────────────
# SIDEBAR — LEVIERS
# ─────────────────────────────────────────────────────────────
tk.sidebar.header("🛠️ Configuration des Leviers")

if "investissements" not in tk.session_state:
    tk.session_state.investissements = {n: 0.0 for n in NOMS}
if "entreprises" not in tk.session_state:
    tk.session_state.entreprises = {n: 10.0 for n in NOMS}
    tk.session_state.entreprises[NOMS[0]] = 65.0

choix_pole = tk.sidebar.selectbox("Pôle à configurer :", NOMS)
tk.sidebar.subheader(f"Paramètres pour : {choix_pole}")

nouveau_inv = tk.sidebar.slider("Budget Infrastructures (Mds FCFA) :", 0, 500,
                                int(tk.session_state.investissements.get(choix_pole, 0)), step=10)
nouveau_ent = tk.sidebar.slider("Concentration Emplois Privés (%) :", 0, 100,
                                int(tk.session_state.entreprises.get(choix_pole, 10)), step=5)
tk.session_state.investissements[choix_pole] = nouveau_inv
tk.session_state.entreprises[choix_pole]     = nouveau_ent

tk.sidebar.subheader("📈 Variables Démographiques")
isf_global     = tk.sidebar.slider("Indice de Fécondité National :", 2.0, 6.5, 4.3, step=0.1)
horizon_annees = tk.sidebar.slider(f"Horizon (années depuis {A1}) :", 0, 15,
                                   min(A2 - A1, 7))

# ─────────────────────────────────────────────────────────────
# MOTEUR DE SIMULATION AVEC LEVIERS
# ─────────────────────────────────────────────────────────────
def executer_simulation_leviers() -> np.ndarray:
    """Applique les leviers d'investissement sur l'attractivité avant simulation."""
    inv_array = np.array([tk.session_state.investissements.get(n, 0) for n in NOMS])
    ent_array = np.array([tk.session_state.entreprises.get(n, 10) for n in NOMS])

    prop = np.array([0.382, 0.592, 0.026])
    X = np.outer(POP_INIT, prop)
    n = N_POLES

    alpha  = params_actifs["alpha"]
    beta   = params_actifs["beta"]
    gamma  = params_actifs["gamma"]
    expo   = params_actifs["expo_dist"]
    t_emig = params_actifs["taux_emig"]
    prime  = params_actifs["prime_cap"]

    for _ in range(horizon_annees):
        isf_loc = isf_global - 1.4 * (1.0 - np.exp(-0.003 * (inv_array + ent_array * 5.0)))
        f = (isf_loc * 0.5) / 35.0

        Xn = np.zeros_like(X)
        Xn[:, 0] = X[:, 0] + X[:, 1] * f      - X[:, 0] * alpha
        Xn[:, 1] = X[:, 1] + X[:, 0] * alpha  - X[:, 1] * beta
        Xn[:, 2] = X[:, 2] + X[:, 1] * beta   - X[:, 2] * gamma

        attractivite = (0.02 / (1.0 + np.exp(-0.012 * (inv_array - 100.0)))) + (0.03 * ent_array / 100.0)
        attractivite[0] += prime

        actifs = Xn[:, 1]
        flux = (actifs[None, :] * attractivite[:, None]) / (MATRICE_DISTANCES ** expo)
        np.fill_diagonal(flux, 0)

        sorties = flux.sum(axis=0)
        for j in range(n):
            lim = actifs[j] * t_emig
            if sorties[j] > lim > 0:
                flux[:, j] *= lim / sorties[j]

        Xn[:, 1] += flux.sum(axis=1) - flux.sum(axis=0)
        X = Xn

    return X.sum(axis=1)


pop_predite = executer_simulation_leviers()

# ─────────────────────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────────────────────
col_g, col_d = tk.columns([2, 1])

with col_g:
    tk.subheader(f"🗺️ Répartition Spatiale en l'An +{horizon_annees} (Base {A1})")

    df_map = pd.DataFrame({
        "Ville": NOMS, "Latitude": LATS, "Longitude": LONS,
        "Population": pop_predite,
        "Affichage": [f"{n} : {int(p):,} hab" for n, p in zip(NOMS, pop_predite)]
    })

    fig = px.scatter_mapbox(
        df_map, lat="Latitude", lon="Longitude",
        size="Population", color="Population",
        color_continuous_scale="Viridis", text="Affichage",
        size_max=50, zoom=5.8,
        center={"lat": 7.5, "lon": -5.5},
        mapbox_style="carto-positron"
    )
    fig.update_traces(textposition="bottom center")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=650)
    tk.plotly_chart(fig, use_container_width=True)

with col_d:
    tk.subheader(f"📊 Validation croisée (Recensement {A2})")

    df_table = pd.DataFrame({
        "Pôle": NOMS,
        f"Simulé (An +{horizon_annees})": pop_predite.astype(int),
        f"Réel {A2}": POP_REEL.astype(int)
    })

    if horizon_annees == (A2 - A1):
        df_table["Écart (%)"] = np.round(
            ((df_table[f"Simulé (An +{horizon_annees})"] - df_table[f"Réel {A2}"]) / df_table[f"Réel {A2}"]) * 100, 1
        )
        tk.success(f"💡 Horizon réglé sur {A2 - A1} ans ({A2}). Comparaison avec le recensement réel.")
    else:
        tk.info(f"Réglez l'horizon sur **{A2 - A1} ans** pour la validation sur le recensement {A2}.")

    tk.dataframe(df_table.set_index("Pôle"), height=550)
