"""
Module d'ingestion et calibration automatique — Simulateur PND Démographique
Office National de la Population (ONP) — Côte d'Ivoire
"""

import streamlit as tk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# GABARIT CSV / EXCEL POUR L'ONP
# ─────────────────────────────────────────────────────────────
COLONNES_REQUISES = ["nom", "latitude", "longitude", "pop_recensement_1", "pop_recensement_2"]
COLONNES_OPTIONNELLES = ["annee_recensement_1", "annee_recensement_2", "region", "statut_administratif"]

GABARIT_DESCRIPTION = """
**Format attendu (CSV ou Excel) :**
| Colonne | Obligatoire | Description |
|---|---|---|
| `nom` | ✅ | Nom du pôle urbain |
| `latitude` | ✅ | Latitude (décimale) |
| `longitude` | ✅ | Longitude (décimale) |
| `pop_recensement_1` | ✅ | Population lors du 1er recensement (période de base) |
| `pop_recensement_2` | ✅ | Population lors du 2e recensement (période de validation) |
| `annee_recensement_1` | ☑️ | Année du 1er recensement (défaut: 2014) |
| `annee_recensement_2` | ☑️ | Année du 2e recensement (défaut: 2021) |
| `region` | ☑️ | Région administrative (optionnel, pour les filtres) |
| `statut_administratif` | ☑️ | Ex: Chef-lieu de région, commune, etc. |
"""

# ─────────────────────────────────────────────────────────────
# DONNÉES INTÉGRÉES (RGPH 2014 & 2021) — FALLBACK OFFICIEL
# ─────────────────────────────────────────────────────────────
DONNEES_INTEGREES = [
    ["Abidjan", 5.348, -4.024, 4395243, 5616633, "Lagunes", "District Autonome"],
    ["Bouaké", 7.690, -5.026, 536719, 728733, "Vallée du Bandama", "Chef-lieu de région"],
    ["Yamoussoukro", 6.820, -5.276, 212670, 422072, "Lacs", "District Autonome"],
    ["Korhogo", 9.458, -5.629, 258699, 386586, "Poro", "Chef-lieu de région"],
    ["Daloa", 6.877, -6.450, 255354, 421879, "Haut-Sassandra", "Chef-lieu de région"],
    ["San-Pédro", 4.748, -6.636, 185623, 390654, "San-Pédro", "Chef-lieu de région"],
    ["Soubré", 5.787, -6.608, 175000, 272773, "Nawa", "Chef-lieu de région"],
    ["Duékoué", 6.742, -7.351, 166633, 220953, "Guémon", "Chef-lieu de région"],
    ["Divo", 5.837, -5.357, 155000, 215312, "Lôh-Djiboua", "Chef-lieu de région"],
    ["Bouaflé", 6.984, -5.752, 145000, 213967, "Marahoué", "Chef-lieu de région"],
    ["Gagnoa", 6.132, -5.951, 168559, 212489, "Gôh", "Chef-lieu de région"],
    ["Bingerville", 5.355, -3.885, 91319, 204656, "Lagunes", "Commune"],
    ["Guiglo", 6.543, -7.494, 125000, 171454, "Cavally", "Chef-lieu de région"],
    ["Lakota", 5.848, -5.682, 120000, 169330, "Lôh-Djiboua", "Commune"],
    ["Ferkessédougou", 9.593, -5.195, 58601, 160267, "Hambol", "Commune"],
    ["Adzopé", 6.107, -3.862, 115000, 156488, "La Mé", "Chef-lieu de région"],
    ["Bondoukou", 8.040, -2.800, 100669, 141568, "Gontougo", "Chef-lieu de région"],
    ["Dabou", 5.326, -4.377, 70773, 138083, "Grands-Ponts", "Chef-lieu de région"],
    ["Sinfra", 6.621, -5.911, 90711, 137210, "Marahoué", "Commune"],
    ["Agboville", 5.928, -4.218, 65982, 135082, "Agnéby-Tiassa", "Chef-lieu de région"],
    ["Oumé", 6.383, -5.417, 85000, 127153, "Gôh", "Commune"],
    ["Abengourou", 6.727, -3.496, 100910, 125143, "Indénié-Djuablin", "Chef-lieu de région"],
    ["Grand-Bassam", 5.212, -3.739, 84028, 124567, "Sud-Comoé", "Chef-lieu de région"],
    ["Séguéla", 7.960, -6.675, 63774, 103980, "Worodougou", "Chef-lieu de région"],
    ["Aboisso", 5.466, -3.207, 75000, 100903, "Sud-Comoé", "Chef-lieu de région"],
    ["Bouna", 9.269, -3.004, 23191, 94883, "Bounkani", "Chef-lieu de région"],
    ["Boundiali", 9.521, -6.486, 43101, 92792, "Bagoué", "Chef-lieu de région"],
    ["Katiola", 8.137, -5.101, 56681, 90641, "Hambol", "Chef-lieu de région"],
    ["Odienné", 9.505, -7.564, 55000, 86279, "Kabadougou", "Chef-lieu de région"],
    ["Vavoua", 7.382, -6.478, 95889, 132528, "Haut-Sassandra", "Commune"],
    ["Danané", 7.260, -8.154, 82290, 131586, "Tonkpi", "Chef-lieu de région"],
    ["Issia", 6.492, -6.586, 63977, 126252, "Haut-Sassandra", "Commune"],
    ["Agni_Bilékrou", 7.111, -3.491, 60000, 82100, "Indénié-Djuablin", "Commune"],
    ["Tanda", 7.803, -3.168, 37989, 77600, "Gontougo", "Commune"],
    ["Tingréla", 10.483, -6.383, 50094, 74200, "Bagoué", "Commune"],
    ["Dimbokro", 6.647, -4.705, 63219, 71200, "Iffou", "Chef-lieu de région"],
    ["Zuénoula", 7.425, -6.050, 52449, 70500, "Marahoué", "Commune"],
    ["Bongouanou", 6.651, -4.304, 62991, 68900, "Moronou", "Chef-lieu de région"],
    ["Touba", 8.283, -7.683, 26310, 65300, "Bafing", "Chef-lieu de région"],
    ["Sassandra", 4.953, -6.083, 45000, 62100, "Gbôklé", "Chef-lieu de région"],
    ["Guintéguéla", 7.912, -7.114, 35000, 58400, "Béré", "Commune"],
    ["Adiaké", 5.286, -3.304, 38000, 55200, "Sud-Comoé", "Commune"],
    ["Bangolo", 7.012, -7.485, 40220, 53100, "Guémon", "Commune"],
    ["Anyama", 5.494, -4.051, 119514, 180000, "Lagunes", "Commune"],
]

def charger_donnees_integrees() -> pd.DataFrame:
    """Retourne le jeu de données officiel RGPH 2014/2021."""
    cols = COLONNES_REQUISES + ["region", "statut_administratif"]
    df = pd.DataFrame(DONNEES_INTEGREES, columns=
                      ["nom","latitude","longitude","pop_recensement_1","pop_recensement_2","region","statut_administratif"])
    df["annee_recensement_1"] = 2014
    df["annee_recensement_2"] = 2021
    return df


def valider_fichier(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Valide le format d'un fichier importé. Retourne (valide, liste_erreurs)."""
    erreurs = []
    for col in COLONNES_REQUISES:
        if col not in df.columns:
            erreurs.append(f"Colonne obligatoire manquante : `{col}`")

    if erreurs:
        return False, erreurs

    # Vérifications numériques
    for col in ["latitude", "longitude", "pop_recensement_1", "pop_recensement_2"]:
        if col in df.columns:
            if not pd.to_numeric(df[col], errors="coerce").notna().all():
                erreurs.append(f"Valeurs non numériques dans la colonne `{col}`")

    # Bornes géographiques (Côte d'Ivoire : lat 4–11, lon -9–-2)
    if "latitude" in df.columns and "longitude" in df.columns:
        lats = pd.to_numeric(df["latitude"], errors="coerce")
        lons = pd.to_numeric(df["longitude"], errors="coerce")
        if not ((lats >= 4) & (lats <= 11)).all():
            erreurs.append("Certaines latitudes semblent hors du territoire (attendu : 4° à 11° N)")
        if not ((lons >= -9) & (lons <= -2)).all():
            erreurs.append("Certaines longitudes semblent hors du territoire (attendu : -9° à -2° E)")

    # Populations positives
    for col in ["pop_recensement_1", "pop_recensement_2"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if (vals <= 0).any():
                erreurs.append(f"Des populations nulles ou négatives dans `{col}`")

    # Doublons
    if df["nom"].duplicated().any():
        doublons = df["nom"][df["nom"].duplicated()].tolist()
        erreurs.append(f"Noms de pôles en doublon : {doublons}")

    return len(erreurs) == 0, erreurs


def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise et complète les colonnes optionnelles."""
    df = df.copy()
    for col in ["latitude", "longitude", "pop_recensement_1", "pop_recensement_2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "annee_recensement_1" not in df.columns:
        df["annee_recensement_1"] = 2014
    if "annee_recensement_2" not in df.columns:
        df["annee_recensement_2"] = 2021
    if "region" not in df.columns:
        df["region"] = "Non renseigné"
    if "statut_administratif" not in df.columns:
        df["statut_administratif"] = "Non renseigné"

    df = df.dropna(subset=COLONNES_REQUISES)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# MOTEUR GRAVITAIRE (VECTORISÉ) + CALIBRATION AUTOMATIQUE
# ─────────────────────────────────────────────────────────────
def calculer_matrice_distances(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    R = 6371.0
    lr, lo = np.radians(lats), np.radians(lons)
    dlat = lr[:, None] - lr[None, :]
    dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lr[:, None]) * np.cos(lr[None, :]) * np.sin(dlon/2)**2
    dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    np.fill_diagonal(dist, 1.0)
    return dist


def executer_modele(pop_init: np.ndarray, distances: np.ndarray,
                    n_annees: int, isf: float, parametres: dict) -> np.ndarray:
    """
    Moteur de projection gravitaire à somme nulle, paramétré.

    parametres : dict avec clés
        alpha       — taux de sortie de la cohorte Jeunes → Actifs
        beta        — taux de sortie de la cohorte Actifs → Seniors
        gamma       — taux de mortalité Seniors
        expo_dist   — exposant de distance dans la loi gravitaire
        taux_emig   — taux max d'émigration annuelle par pôle
        prime_cap   — prime d'attractivité structurelle du 1er pôle (Abidjan)
    """
    alpha   = parametres.get("alpha",     1/15)
    beta    = parametres.get("beta",      1/50)
    gamma   = parametres.get("gamma",     0.075)
    expo    = parametres.get("expo_dist", 1.5)
    t_emig  = parametres.get("taux_emig", 0.03)
    prime   = parametres.get("prime_cap", 0.015)

    n = len(pop_init)
    prop = np.array([0.382, 0.592, 0.026])
    X = np.outer(pop_init, prop)   # [n, 3]

    # Attractivité neutre (sans investissements, pour calibration pure)
    attractivite = np.ones(n) * 0.025
    attractivite[0] += prime

    for _ in range(n_annees):
        isf_loc = isf - 1.4 * 0.3   # effet moyen sans leviers
        f = (isf_loc * 0.5) / 35.0

        Xn = np.zeros_like(X)
        Xn[:, 0] = X[:, 0] + X[:, 1] * f       - X[:, 0] * alpha
        Xn[:, 1] = X[:, 1] + X[:, 0] * alpha   - X[:, 1] * beta
        Xn[:, 2] = X[:, 2] + X[:, 1] * beta    - X[:, 2] * gamma

        # Flux gravitaires
        actifs = Xn[:, 1]
        flux = (actifs[None, :] * attractivite[:, None]) / (distances ** expo)
        np.fill_diagonal(flux, 0)

        sorties = flux.sum(axis=0)
        for j in range(n):
            lim = actifs[j] * t_emig
            if sorties[j] > lim > 0:
                flux[:, j] *= lim / sorties[j]

        Xn[:, 1] += flux.sum(axis=1) - flux.sum(axis=0)
        X = Xn

    return X.sum(axis=1)


def calibrer_parametres(df: pd.DataFrame, isf: float = 4.3) -> tuple[dict, pd.DataFrame]:
    """
    Calibre automatiquement les 6 paramètres du modèle par minimisation
    de l'erreur quadratique moyenne sur l'historique intercensitaire.

    Retourne (parametres_calibres, df_calibration_detail).
    """
    pop1 = df["pop_recensement_1"].values.astype(float)
    pop2 = df["pop_recensement_2"].values.astype(float)
    n_annees = int(df["annee_recensement_2"].iloc[0]) - int(df["annee_recensement_1"].iloc[0])
    distances = calculer_matrice_distances(df["latitude"].values, df["longitude"].values)

    # Paramètres initiaux et bornes
    x0     = [1/15,  1/50,  0.075, 1.5,  0.03, 0.015]
    bornes = [(1/25, 1/10), (1/80, 1/30), (0.02, 0.15),
              (0.8,  2.5),  (0.005, 0.08), (0.0, 0.05)]

    def objectif(x):
        params = dict(zip(["alpha","beta","gamma","expo_dist","taux_emig","prime_cap"], x))
        pred = executer_modele(pop1, distances, n_annees, isf, params)
        # RMSE normalisé (relatif)
        erreurs_rel = ((pred - pop2) / pop2) ** 2
        return np.sqrt(erreurs_rel.mean()) * 100

    resultat = minimize(objectif, x0, bounds=bornes, method="L-BFGS-B",
                        options={"maxiter": 300, "ftol": 1e-9})

    params_calibres = dict(zip(["alpha","beta","gamma","expo_dist","taux_emig","prime_cap"],
                               resultat.x))

    # Calcul détaillé post-calibration
    pred_calibree = executer_modele(pop1, distances, n_annees, isf, params_calibres)
    ecart_rel = ((pred_calibree - pop2) / pop2) * 100

    df_detail = pd.DataFrame({
        "Pôle":              df["nom"].values,
        "Région":            df["region"].values,
        "Pop. Recensement 1": pop1.astype(int),
        "Pop. Recensement 2 (Réel)": pop2.astype(int),
        "Pop. Simulée (Calibrée)": pred_calibree.astype(int),
        "Écart Relatif (%)": np.round(ecart_rel, 2),
        "Écart Absolu":      (pred_calibree - pop2).astype(int),
    })

    return params_calibres, df_detail


def generer_gabarit_csv(vide: bool = False) -> bytes:
    """Génère un gabarit CSV pré-rempli ou vide pour l'ONP."""
    if vide:
        df = pd.DataFrame(columns=COLONNES_REQUISES + COLONNES_OPTIONNELLES)
        df.loc[0] = ["Exemple_Ville", 6.5, -5.2, 100000, 130000, 2014, 2021, "Région X", "Chef-lieu"]
    else:
        df = charger_donnees_integrees()
    return df.to_csv(index=False).encode("utf-8-sig")   # BOM pour Excel FR


def generer_gabarit_excel(vide: bool = False) -> bytes:
    """Génère un gabarit Excel avec instructions."""
    df_data = pd.DataFrame(columns=COLONNES_REQUISES + COLONNES_OPTIONNELLES) if vide else charger_donnees_integrees()
    instructions = pd.DataFrame({
        "Colonne": COLONNES_REQUISES + COLONNES_OPTIONNELLES,
        "Obligatoire": ["✅"]*5 + ["☑️"]*4,
        "Description": [
            "Nom du pôle urbain ou rural",
            "Latitude en degrés décimaux (ex: 6.820)",
            "Longitude en degrés décimaux (ex: -5.276)",
            "Population lors du premier recensement",
            "Population lors du deuxième recensement",
            "Année du 1er recensement (si vide : 2014)",
            "Année du 2e recensement (si vide : 2021)",
            "Région administrative (optionnel)",
            "Statut administratif (optionnel)",
        ]
    })

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_data.to_excel(writer, sheet_name="Données_Pôles", index=False)
        instructions.to_excel(writer, sheet_name="📋 Instructions", index=False)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# PAGE STREAMLIT : MODULE D'INGESTION ONP
# ─────────────────────────────────────────────────────────────
def afficher_page_ingestion():
    """Page Streamlit complète pour l'ingestion et la calibration."""
    tk.title("📥 Module d'Ingestion — Données ONP")
    tk.caption("Importez vos propres fichiers de recensement ou travaillez avec les données RGPH officielles intégrées.")

    # ── Onglets principaux ─────────────────────────────────
    tab_source, tab_gabarit, tab_calibration, tab_export = tk.tabs([
        "1. Source de données", "2. Gabarits ONP", "3. Calibration automatique", "4. Export"
    ])

    # ── TAB 1 : Source ────────────────────────────────────
    with tab_source:
        tk.subheader("Source des données démographiques")

        source = tk.radio(
            "Choisissez votre source :",
            ["🏛️ Données intégrées RGPH 2014 & 2021 (officiel)",
             "📂 Importer un fichier CSV / Excel (données ONP)"],
            horizontal=True
        )

        if source.startswith("🏛️"):
            df = charger_donnees_integrees()
            tk.success(f"✅ {len(df)} pôles chargés depuis les données RGPH officielles.")

        else:
            fichier = tk.file_uploader(
                "Déposez votre fichier ici :",
                type=["csv", "xlsx", "xls"],
                help="Format attendu : voir l'onglet 'Gabarits ONP'"
            )

            if fichier is None:
                tk.info("💡 Aucun fichier sélectionné. Téléchargez d'abord un gabarit depuis l'onglet **Gabarits ONP**.")
                df = charger_donnees_integrees()
            else:
                try:
                    if fichier.name.endswith(".csv"):
                        df_brut = pd.read_csv(fichier)
                    else:
                        df_brut = pd.read_excel(fichier, sheet_name=0)

                    valide, erreurs = valider_fichier(df_brut)

                    if not valide:
                        tk.error("❌ Erreurs détectées dans le fichier :")
                        for e in erreurs:
                            tk.markdown(f"- {e}")
                        tk.markdown(GABARIT_DESCRIPTION)
                        df = charger_donnees_integrees()
                        tk.warning("Données RGPH intégrées chargées par défaut.")
                    else:
                        df = nettoyer_donnees(df_brut)
                        tk.success(f"✅ Fichier valide — {len(df)} pôles importés avec succès.")

                except Exception as exc:
                    tk.error(f"Erreur de lecture : {exc}")
                    df = charger_donnees_integrees()

        # Stockage dans session_state pour partage entre tabs
        tk.session_state["df_onp"] = df

        # Aperçu
        with tk.expander("👁️ Aperçu des données chargées", expanded=True):
            col1, col2, col3 = tk.columns(3)
            a1 = df["annee_recensement_1"].iloc[0] if "annee_recensement_1" in df.columns else 2014
            a2 = df["annee_recensement_2"].iloc[0] if "annee_recensement_2" in df.columns else 2021
            col1.metric("Nombre de pôles", len(df))
            col2.metric(f"Pop. totale recensement {a1}", f"{df['pop_recensement_1'].sum():,.0f}".replace(",", " "))
            col3.metric(f"Pop. totale recensement {a2}", f"{df['pop_recensement_2'].sum():,.0f}".replace(",", " "))

            tk.dataframe(df, use_container_width=True, height=300)

            # Carte rapide
            fig_prev = px.scatter_mapbox(
                df, lat="latitude", lon="longitude",
                size="pop_recensement_2", color="pop_recensement_2",
                hover_name="nom", hover_data={"pop_recensement_1": True, "pop_recensement_2": True},
                color_continuous_scale="Viridis", size_max=40, zoom=5.5,
                center={"lat": 7.5, "lon": -5.5}, mapbox_style="carto-positron",
                title=f"Distribution spatiale — Recensement {a2}"
            )
            fig_prev.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, height=400)
            tk.plotly_chart(fig_prev, use_container_width=True)

    # ── TAB 2 : Gabarits ─────────────────────────────────
    with tab_gabarit:
        tk.subheader("📄 Gabarits téléchargeables")
        tk.markdown(GABARIT_DESCRIPTION)

        col_g1, col_g2, col_g3, col_g4 = tk.columns(4)

        with col_g1:
            tk.download_button(
                "⬇️ Gabarit CSV (vide)",
                data=generer_gabarit_csv(vide=True),
                file_name="gabarit_ONP_vide.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_g2:
            tk.download_button(
                "⬇️ Gabarit Excel (vide)",
                data=generer_gabarit_excel(vide=True),
                file_name="gabarit_ONP_vide.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with col_g3:
            tk.download_button(
                "⬇️ CSV pré-rempli RGPH 2014/2021",
                data=generer_gabarit_csv(vide=False),
                file_name="donnees_RGPH_2014_2021.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_g4:
            tk.download_button(
                "⬇️ Excel pré-rempli RGPH 2014/2021",
                data=generer_gabarit_excel(vide=False),
                file_name="donnees_RGPH_2014_2021.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        tk.info("💡 **Conseils pour l'ONP :** Téléchargez le gabarit Excel pré-rempli pour voir le format attendu, "
                "puis remplacez ou complétez les données avec les nouvelles collectes.")

    # ── TAB 3 : Calibration ──────────────────────────────
    with tab_calibration:
        tk.subheader("⚙️ Calibration automatique des paramètres du modèle")
        tk.markdown("""
        Le moteur gravitaire contient **6 paramètres libres**. La calibration les ajuste automatiquement 
        pour minimiser l'erreur entre les populations simulées et les recensements réels.
        Cette étape est **indispensable** avant toute projection.
        """)

        df_cal = tk.session_state.get("df_onp", charger_donnees_integrees())
        a1 = int(df_cal["annee_recensement_1"].iloc[0]) if "annee_recensement_1" in df_cal.columns else 2014
        a2 = int(df_cal["annee_recensement_2"].iloc[0]) if "annee_recensement_2" in df_cal.columns else 2021

        col_c1, col_c2 = tk.columns([1, 2])
        with col_c1:
            isf_cal = tk.slider("ISF moyen de la période intercensitaire :", 3.5, 6.0, 4.7, 0.1)
            lancer = tk.button("🚀 Lancer la calibration", type="primary", use_container_width=True)

        if lancer:
            with tk.spinner(f"Calibration en cours sur {a2 - a1} années ({a1}→{a2})…"):
                try:
                    params, df_detail = calibrer_parametres(df_cal, isf=isf_cal)
                    tk.session_state["params_calibres"] = params
                    tk.session_state["df_calibration_detail"] = df_detail
                    tk.success("✅ Calibration terminée avec succès !")
                except Exception as exc:
                    tk.error(f"Erreur lors de la calibration : {exc}")

        if "params_calibres" in tk.session_state:
            params = tk.session_state["params_calibres"]
            df_detail = tk.session_state["df_calibration_detail"]

            # Résumé des paramètres
            tk.subheader("Paramètres calibrés")
            cp1, cp2, cp3, cp4, cp5, cp6 = tk.columns(6)
            cp1.metric("α (Jeunes→Actifs)", f"1/{1/params['alpha']:.0f}")
            cp2.metric("β (Actifs→Seniors)", f"1/{1/params['beta']:.0f}")
            cp3.metric("γ (Mortalité Seniors)", f"{params['gamma']:.3f}")
            cp4.metric("Exposant Distance", f"{params['expo_dist']:.2f}")
            cp5.metric("Taux Max Émigration", f"{params['taux_emig']*100:.1f}%")
            cp6.metric("Prime Capitale", f"{params['prime_cap']*100:.2f}%")

            # Qualité de la calibration
            rmse = np.sqrt((df_detail["Écart Relatif (%)"]**2).mean())
            mae  = df_detail["Écart Relatif (%)"].abs().mean()
            tk.subheader("Qualité du modèle calibré")
            q1, q2, q3 = tk.columns(3)
            q1.metric("RMSE Relatif", f"{rmse:.2f}%", help="Racine de l'erreur quadratique moyenne relative")
            q2.metric("MAE Relative", f"{mae:.2f}%", help="Erreur absolue moyenne relative")
            q3.metric("Pôles dans ±10%", f"{(df_detail['Écart Relatif (%)'].abs() < 10).sum()}/{len(df_detail)}")

            # Graphique erreurs
            fig_err = px.bar(
                df_detail.sort_values("Écart Relatif (%)", key=abs, ascending=False).head(20),
                x="Pôle", y="Écart Relatif (%)",
                color="Écart Relatif (%)", color_continuous_scale="RdYlGn_r",
                title="Top 20 pôles par écart relatif (simulation calibrée vs recensement réel)"
            )
            fig_err.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="+10%")
            fig_err.add_hline(y=-10, line_dash="dash", line_color="orange", annotation_text="-10%")
            fig_err.update_layout(height=380)
            tk.plotly_chart(fig_err, use_container_width=True)

            # Scatter simulé vs réel
            fig_sc = px.scatter(
                df_detail, x="Pop. Recensement 2 (Réel)", y="Pop. Simulée (Calibrée)",
                hover_name="Pôle", size="Pop. Recensement 1",
                title="Validation : Population simulée vs recensement réel",
                labels={"Pop. Recensement 2 (Réel)": f"Réel RGPH {a2}",
                        "Pop. Simulée (Calibrée)": f"Simulé {a2}"}
            )
            max_val = max(df_detail["Pop. Recensement 2 (Réel)"].max(),
                          df_detail["Pop. Simulée (Calibrée)"].max()) * 1.05
            fig_sc.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                        mode="lines", name="Parfait accord",
                                        line=dict(dash="dot", color="gray")))
            fig_sc.update_layout(height=420)
            tk.plotly_chart(fig_sc, use_container_width=True)

            # Tableau détaillé
            with tk.expander("📋 Tableau détaillé de calibration"):
                tk.dataframe(
                    df_detail.set_index("Pôle").style.background_gradient(
                        subset=["Écart Relatif (%)"], cmap="RdYlGn_r", vmin=-20, vmax=20
                    ),
                    use_container_width=True
                )

    # ── TAB 4 : Export ───────────────────────────────────
    with tab_export:
        tk.subheader("📤 Export des données et résultats")

        df_exp = tk.session_state.get("df_onp", charger_donnees_integrees())

        col_e1, col_e2, col_e3 = tk.columns(3)

        with col_e1:
            tk.markdown("**Données sources nettoyées**")
            tk.download_button(
                "⬇️ Données ONP (CSV)",
                data=df_exp.to_csv(index=False).encode("utf-8-sig"),
                file_name="donnees_onp_validees.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_e2:
            if "params_calibres" in tk.session_state:
                tk.markdown("**Paramètres calibrés**")
                params_df = pd.DataFrame([tk.session_state["params_calibres"]])
                tk.download_button(
                    "⬇️ Paramètres calibrés (CSV)",
                    data=params_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="parametres_calibration_onp.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                tk.markdown("**Paramètres calibrés**")
                tk.caption("Lancez la calibration d'abord (onglet 3).")

        with col_e3:
            if "df_calibration_detail" in tk.session_state:
                tk.markdown("**Rapport de validation**")
                buf = BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df_exp.to_excel(writer, sheet_name="Données_Sources", index=False)
                    tk.session_state["df_calibration_detail"].to_excel(
                        writer, sheet_name="Calibration_Validation", index=False)
                    if "params_calibres" in tk.session_state:
                        pd.DataFrame([tk.session_state["params_calibres"]]).to_excel(
                            writer, sheet_name="Paramètres", index=False)

                tk.download_button(
                    "⬇️ Rapport complet (Excel)",
                    data=buf.getvalue(),
                    file_name="rapport_calibration_onp.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                tk.markdown("**Rapport de validation**")
                tk.caption("Lancez la calibration d'abord (onglet 3).")

        tk.divider()
        tk.info(
            "💡 **Workflow recommandé ONP :**\n"
            "1. Importez vos données dans l'onglet **Source de données**\n"
            "2. Vérifiez la carte et l'aperçu\n"
            "3. Lancez la **calibration automatique**\n"
            "4. Exportez le rapport Excel comme pièce justificative méthodologique\n"
            "5. Les paramètres calibrés sont automatiquement transmis au simulateur principal"
        )


# ─────────────────────────────────────────────────────────────
# POINT D'ENTRÉE (MODULE STANDALONE OU INTÉGRATION)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    afficher_page_ingestion()
