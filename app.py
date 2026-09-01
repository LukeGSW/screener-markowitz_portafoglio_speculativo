"""
Screener & Frontiera Efficiente - Scuola di Finanza Operativa

Corso 2: Portafogli Avanzati.
Conversione in applicazione web del notebook Colab del corso.

Avvio in locale:  streamlit run app.py
Dati:             archivio Parquet in data/, costruito da scripts/build_dataset.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from src import charts, datastore, portfolio, report, screener, texts
from src.config import (APP_COURSE, APP_SUBTITLE, APP_TITLE, APP_VERSION,
                        ARCHIVE_MAX_AGE_DAYS, BENCHMARK, COLORS,
                        DEFAULT_MAX_DRAWDOWN, DEFAULT_MAX_PRICE,
                        DEFAULT_MAX_TICKERS, DEFAULT_MIN_DOLLAR_VOLUME,
                        DEFAULT_MIN_PRICE, DEFAULT_MIN_SHARPE,
                        DEFAULT_MIN_TICKERS, DEFAULT_MIN_VOL,
                        DEFAULT_NUM_SIMULATIONS, DEFAULT_REBALANCE,
                        DEFAULT_RISK_FREE, DEFAULT_SEED,
                        DEFAULT_YEARS_HISTORY, DEFAULT_YEARS_NO_NEG,
                        FRONTIER_MAX_POINTS, HEATMAP_MAX_TICKERS,
                        HEATMAP_TOP_PORTFOLIOS, LIQUIDITY_LOOKBACK,
                        MIN_TICKERS_FOR_MAX_RETURN, NAME_BENCHMARK,
                        REBALANCE_METHODS, REBALANCE_ORDER, ROLLING_WINDOWS,
                        STRATEGY_COLORS)

st.set_page_config(
    page_title=f"{APP_TITLE} | Kriterion Quant",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1550px; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.45rem; }}
      div[data-testid="stMetricLabel"] {{ color: {COLORS['muted']}; }}
      .kq-hero {{
          background: linear-gradient(135deg, #101B2E 0%, #0B1220 100%);
          border: 1px solid {COLORS['grid']}; border-left: 4px solid {COLORS['accent']};
          border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem;
      }}
      .kq-hero h1 {{ font-size: 1.7rem; margin: 0 0 .25rem 0; color: {COLORS['text']}; }}
      .kq-hero p {{ margin: 0; color: {COLORS['muted']}; font-size: .92rem; }}
      .kq-note {{
          background: {COLORS['panel']}; border-left: 3px solid {COLORS['accent']};
          border-radius: 6px; padding: .8rem 1rem; margin: .6rem 0;
          font-size: .9rem; color: {COLORS['text']};
      }}
      .kq-warn {{
          background: rgba(251, 191, 36, .09); border-left: 3px solid {COLORS['warning']};
          border-radius: 6px; padding: .8rem 1rem; margin: .6rem 0; font-size: .9rem;
      }}
      section[data-testid="stSidebar"] {{ width: 400px !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Formattazione all'italiana
# ==========================================================================
def _it(valore: float, decimali: int) -> str:
    s = f"{valore:,.{decimali}f}"
    return s.translate(str.maketrans({",": ".", ".": ","}))


def num(valore, decimali: int = 0) -> str:
    if valore is None or not np.isfinite(valore):
        return "n.d."
    return _it(float(valore), decimali)


def pct(valore, decimali: int = 2) -> str:
    if valore is None or not np.isfinite(valore):
        return "n.d."
    return f"{_it(float(valore) * 100.0, decimali)}%"


def dollari(valore, decimali: int = 0) -> str:
    if valore is None or not np.isfinite(valore):
        return "n.d."
    return f"$ {_it(float(valore), decimali)}"


def nota(testo: str) -> None:
    st.markdown(f'<div class="kq-note">{testo}</div>', unsafe_allow_html=True)


def avviso(testo: str) -> None:
    st.markdown(f'<div class="kq-warn">{testo}</div>', unsafe_allow_html=True)


# ==========================================================================
# Accesso all'archivio
# ==========================================================================
@st.cache_data(show_spinner=False)
def carica_metriche(firma: tuple) -> pd.DataFrame:
    """La tabella delle misure. La firma serve solo a invalidare la cache."""
    return datastore.leggi_metriche()


@st.cache_data(show_spinner=False)
def carica_universo(firma: tuple) -> pd.DataFrame:
    return datastore.leggi_universo()


@st.cache_data(show_spinner=False)
def carica_prezzi(tickers: tuple[str, ...], firma: tuple) -> pd.DataFrame:
    mappa = datastore.mappa_fette_da_metriche(carica_metriche(firma))
    return datastore.leggi_prezzi(list(tickers), mappa)


def firma_archivio() -> tuple:
    """Identifica la versione dell'archivio, per la cache di Streamlit."""
    meta = datastore.leggi_meta() or {}
    return (str(meta.get("costruito_il", "")), int(meta.get("n_universo", 0)))


def prepara_archivio() -> bool:
    """
    Assicura che i file leggeri dell'archivio siano presenti.

    Sono meta.json, l'anagrafica e la tabella delle misure: qualche megabyte
    che basta a far funzionare lo screener sull'universo intero. Le fette dei
    prezzi arrivano solo al momento di simulare.
    """
    if datastore.archivio_presente():
        return True

    indirizzo = datastore.indirizzo_archivio()
    if not indirizzo:
        return False

    barra = st.progress(0.0, "Scarico l'archivio...")
    try:
        def avanzamento(nome: str, fatti: int, totale: int) -> None:
            quota = fatti / totale if totale else 0.0
            barra.progress(min(quota, 1.0),
                           f"Scarico {nome}: {fatti / 1024 ** 2:.1f} MB")

        datastore.assicura_nucleo(on_progress=avanzamento)
        barra.empty()
        return True
    except Exception as exc:
        barra.empty()
        st.error(f"Non riesco a scaricare l'archivio: {exc}")
        return False


# ==========================================================================
# Intestazione
# ==========================================================================
st.markdown(
    f"""
    <div class="kq-hero">
      <h1>{APP_TITLE}</h1>
      <p>{APP_SUBTITLE} &nbsp;&middot;&nbsp; {APP_COURSE} &nbsp;&middot;&nbsp; v{APP_VERSION}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

archivio_pronto = prepara_archivio()
meta = datastore.leggi_meta() or {}

if not archivio_pronto:
    st.error("**Archivio dati assente.**")
    st.markdown(
        """
Questa applicazione non scarica i dati mentre la si usa: legge un archivio
costruito in precedenza. Per averlo ci sono due strade.

**In aula o in locale** — costruiscilo una volta, con la tua chiave EODHD:

```bash
export EODHD_API_KEY="la_tua_chiave"
python scripts/build_dataset.py --universo USA --anni 20 --out data
```

Per una prova rapida, senza aspettare mezz'ora, aggiungi `--limite 400`.

**Sull'applicazione pubblicata** — imposta `ARCHIVE_URL` nei secrets di
Streamlit, facendolo puntare alla release di GitHub prodotta dal workflow
`aggiorna-archivio`. L'app scarichera' da sola quel che le serve.
        """
    )
    st.stop()

firma = firma_archivio()
metriche = carica_metriche(firma)
anagrafica = carica_universo(firma)
eta = datastore.eta_giorni(meta)

# ==========================================================================
# Barra laterale: i filtri
# ==========================================================================
with st.sidebar:
    st.header("Filtri dello screener")
    st.caption(
        f"Universo: **{meta.get('n_universo', len(metriche)):,}** titoli"
        .replace(",", ".")
        + f" &middot; archivio del {str(meta.get('costruito_il', ''))[:10]}"
    )

    st.subheader("Storia e qualita'")
    anni_storico = st.slider(
        "Anni di storico richiesti", 1, 30, DEFAULT_YEARS_HISTORY,
        help=texts.FILTRI_HELP["anni_storico"])
    anni_perf = st.slider(
        "Anni senza alcun periodo in perdita", 1, 10, DEFAULT_YEARS_NO_NEG,
        help=texts.FILTRI_HELP["anni_perf_continua"])
    max_dd = st.slider(
        "Drawdown massimo tollerato", 0.05, 1.0, DEFAULT_MAX_DRAWDOWN, 0.05,
        format="%.2f", help=texts.FILTRI_HELP["max_drawdown"])
    min_sharpe = st.slider(
        "Sharpe ratio minimo", -1.0, 3.0, DEFAULT_MIN_SHARPE, 0.1,
        help=texts.FILTRI_HELP["min_sharpe"])
    min_vol = st.slider(
        "Volatilita' minima (mensile annualizzata)", 0.0, 1.5,
        DEFAULT_MIN_VOL, 0.05, help=texts.FILTRI_HELP["min_volatilita"])

    st.subheader("Prezzo e liquidita'")
    prezzo_min = st.number_input(
        "Prezzo minimo (USD)", 0.0, 10_000.0, DEFAULT_MIN_PRICE, 1.0,
        help=texts.FILTRI_HELP["prezzo_minimo"])
    prezzo_max = st.number_input(
        "Prezzo massimo (USD)", 1.0, 1_000_000.0, DEFAULT_MAX_PRICE, 100.0,
        help=texts.FILTRI_HELP["prezzo_massimo"])
    dollar_volume = st.number_input(
        f"Volume medio minimo in dollari ({LIQUIDITY_LOOKBACK} giorni)",
        0.0, 1e10, DEFAULT_MIN_DOLLAR_VOLUME, 500_000.0, format="%.0f",
        help=texts.FILTRI_HELP["dollar_volume_minimo"])

    st.subheader("Simulazione")
    min_titoli = st.slider("Titoli per portafoglio: minimo", 1, 25,
                           DEFAULT_MIN_TICKERS)
    max_titoli = st.slider("Titoli per portafoglio: massimo", 1, 40,
                           DEFAULT_MAX_TICKERS)
    if max_titoli < min_titoli:
        max_titoli = min_titoli
    n_simulazioni = st.select_slider(
        "Portafogli da estrarre",
        options=[5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000],
        value=DEFAULT_NUM_SIMULATIONS)
    ribilanciamento = st.selectbox(
        "Ribilanciamento", REBALANCE_ORDER,
        index=REBALANCE_ORDER.index(DEFAULT_REBALANCE),
        format_func=lambda k: REBALANCE_METHODS[k])
    st.caption(texts.RIBILANCIAMENTO_HELP[ribilanciamento])

    tasso_rf = st.number_input(
        "Tasso privo di rischio annuo", -0.05, 0.20, DEFAULT_RISK_FREE, 0.005,
        format="%.3f", help=texts.FILTRI_HELP["tasso_privo_rischio"])
    seme = st.number_input("Seme casuale", 0, 10_000, DEFAULT_SEED, 1,
                           help="Stesso seme, stessi portafogli. Serve in aula.")

    st.divider()
    st.caption("Dati: EOD Historical Data &middot; strumento didattico, "
               "non consulenza finanziaria.")

filtri = screener.Filtri(
    anni_storico=int(anni_storico),
    anni_perf_continua=int(anni_perf),
    max_drawdown=float(max_dd),
    min_sharpe=float(min_sharpe),
    min_volatilita=float(min_vol),
    prezzo_minimo=float(prezzo_min),
    prezzo_massimo=float(prezzo_max),
    dollar_volume_minimo=float(dollar_volume),
    tasso_privo_rischio=float(tasso_rf),
)

# Lo screening dell'universo intero, a ogni interazione. Costa millisecondi.
risultato = screener.applica(metriche, filtri)
riepilogo = screener.riepilogo(risultato)
titoli_promossi = screener.promossi(risultato)

# ==========================================================================
# Schede
# ==========================================================================
tab_screener, tab_sim, tab_report, tab_dati, tab_metodo = st.tabs([
    "Screener", "Simulazione", "Report", "Dati", "Metodologia",
])


# --------------------------------------------------------------------------
# Scheda: Screener
# --------------------------------------------------------------------------
with tab_screener:
    st.markdown(texts.INTRO)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universo", num(len(metriche)))
    c2.metric("Promossi", num(len(titoli_promossi)),
              f"{len(titoli_promossi) / max(len(metriche), 1) * 100:.2f}% dell'universo")
    con_storia = int((risultato["esito"] != "STORICO").sum()
                     - (risultato["esito"] == "DATI").sum())
    c3.metric(f"Con {filtri.anni_storico} anni di storia", num(max(con_storia, 0)))
    if titoli_promossi:
        c4.metric("Sharpe mediano dei promossi",
                  num(risultato.loc[risultato["esito"] == "PASSATO",
                                    "sharpe"].median(), 2))
    else:
        c4.metric("Sharpe mediano dei promossi", "n.d.")

    if not titoli_promossi:
        avviso("<b>Nessun titolo supera i filtri.</b> Guarda la tabella qui "
               "sotto: la barra piu' lunga e' il vincolo che sta decidendo "
               "tutto, ed e' quello da allentare per primo.")
    elif len(titoli_promossi) < min_titoli:
        avviso(f"<b>Solo {len(titoli_promossi)} titoli promossi</b>, ma i "
               f"portafogli ne richiedono almeno {min_titoli}. Allenta i "
               "filtri o abbassa il numero minimo di titoli.")

    st.divider()
    col_sx, col_dx = st.columns([1.05, 1])

    with col_sx:
        st.subheader("Dove si ferma l'universo")
        st.markdown(texts.IMBUTO)
        st.plotly_chart(charts.imbuto_filtri(riepilogo),
                        width="stretch")

    with col_dx:
        st.subheader("Il conto esatto")
        tabella = riepilogo.copy()
        tabella["Titoli"] = tabella["Titoli"].map(lambda v: num(v))
        tabella["Quota"] = riepilogo["Quota"].map(lambda v: pct(v, 2))
        st.dataframe(tabella, width="stretch", hide_index=True,
                     height=390)

        st.markdown("**Quanto dipende da una sola soglia**")
        quale = st.selectbox(
            "Soglia da far variare",
            ["anni_storico", "anni_perf_continua", "min_sharpe",
             "max_drawdown", "dollar_volume_minimo", "prezzo_minimo"],
            format_func=lambda k: {
                "anni_storico": "Anni di storico",
                "anni_perf_continua": "Anni senza periodi in perdita",
                "min_sharpe": "Sharpe minimo",
                "max_drawdown": "Drawdown massimo",
                "dollar_volume_minimo": "Volume in dollari minimo",
                "prezzo_minimo": "Prezzo minimo",
            }[k], label_visibility="collapsed")

        griglie = {
            "anni_storico": list(range(1, 31)),
            "anni_perf_continua": list(range(1, 11)),
            "min_sharpe": [round(v, 2) for v in np.arange(-0.5, 2.01, 0.1)],
            "max_drawdown": [round(v, 2) for v in np.arange(0.1, 1.01, 0.05)],
            "dollar_volume_minimo": [0, 1e5, 5e5, 1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8],
            "prezzo_minimo": [0, 1, 2, 5, 10, 20, 50, 100],
        }
        curva = screener.curva_sensibilita(metriche, filtri, quale,
                                           griglie[quale])
        st.plotly_chart(
            charts.sensibilita(curva, quale.replace("_", " "),
                               getattr(filtri, quale),
                               percentuale=quale == "max_drawdown"),
            width="stretch")

    st.divider()
    st.subheader("L'universo nel piano rischio-rendimento")
    st.markdown(texts.NUVOLA)
    st.plotly_chart(charts.nuvola_universo(risultato), width="stretch")

    with st.expander("Che cosa fanno gli otto filtri", expanded=False):
        st.markdown(texts.SCREENER)

    st.divider()
    st.subheader(f"I {len(titoli_promossi)} titoli promossi")
    if titoli_promossi:
        dettaglio = screener.dettaglio_promossi(risultato)
        if not anagrafica.empty:
            dettaglio = dettaglio.merge(
                anagrafica[["ticker", "name", "type"]], on="ticker", how="left")
        st.dataframe(
            dettaglio,
            width="stretch", hide_index=True, height=460,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "name": st.column_config.TextColumn("Nome"),
                "type": st.column_config.TextColumn("Tipo", width="small"),
                "anni_storico": st.column_config.NumberColumn(
                    "Anni", format="%.1f", width="small"),
                "prezzo_ultimo": st.column_config.NumberColumn(
                    "Prezzo", format="$ %.2f", width="small"),
                f"dollar_volume_{LIQUIDITY_LOOKBACK}":
                    st.column_config.NumberColumn("Volume $/gg", format="compact"),
                "cagr": st.column_config.NumberColumn("CAGR", format="percent"),
                "volatilita_mensile": st.column_config.NumberColumn(
                    "Vol. mensile", format="percent"),
                "dev_std_ann": st.column_config.NumberColumn(
                    "Vol. giorn.", format="percent"),
                "max_drawdown": st.column_config.NumberColumn(
                    "Drawdown", format="percent"),
                "sharpe": st.column_config.NumberColumn(
                    "Sharpe", format="%.3f", width="small"),
            })
        st.download_button(
            "Scarica l'elenco in CSV",
            dettaglio.to_csv(index=False).encode("utf-8"),
            file_name=f"titoli_promossi_{dt.date.today().isoformat()}.csv",
            mime="text/csv")


# --------------------------------------------------------------------------
# Scheda: Simulazione
# --------------------------------------------------------------------------
with tab_sim:
    st.subheader("Estrazione dei portafogli")
    st.markdown(texts.SIMULAZIONE_HELP)

    abbastanza = len(titoli_promossi) >= max(min_titoli, 2)
    if not abbastanza:
        avviso("Servono almeno due titoli promossi (e almeno quanti ne "
               "richiede il portafoglio minimo) per poter simulare. "
               "Torna alla scheda <b>Screener</b> e allenta i filtri.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("Titoli disponibili", num(len(titoli_promossi)))
        c2.metric("Portafogli da estrarre", num(n_simulazioni))
        lancia = c3.button("Estrai i portafogli", type="primary",
                           width="stretch")

        if lancia:
            with st.spinner("Carico le serie storiche dei titoli promossi..."):
                mappa = datastore.mappa_fette_da_metriche(metriche)
                fette = {mappa[t] for t in titoli_promossi if t in mappa}
                if BENCHMARK in mappa:
                    fette.add(mappa[BENCHMARK])
                try:
                    datastore.assicura_fette(fette)
                except Exception as exc:
                    st.error(f"Non riesco a recuperare le serie storiche: {exc}")
                    st.stop()

                elenco = tuple(sorted(set(titoli_promossi) | {BENCHMARK}))
                prezzi = carica_prezzi(elenco, firma)

            if prezzi.empty:
                st.error("L'archivio non contiene le serie storiche di questi "
                         "titoli. Ricostruiscilo con scripts/build_dataset.py.")
                st.stop()

            with st.spinner("Preparo i rendimenti ed espello i titoli troppo giovani..."):
                universo_sim = portfolio.prepara_universo(
                    prezzi, BENCHMARK, filtri.anni_storico)

            if len(universo_sim.tickers) < min_titoli:
                st.error(
                    f"Dopo il controllo sulla profondita' storica restano "
                    f"{len(universo_sim.tickers)} titoli, meno dei "
                    f"{min_titoli} richiesti.")
                st.stop()

            with st.spinner(f"Estraggo {n_simulazioni:,} portafogli..."):
                mu, cov = portfolio.matrici(universo_sim)
                sim = portfolio.simula(
                    mu, cov, universo_sim.tickers,
                    k_min=min_titoli, k_max=max_titoli,
                    n_simulazioni=int(n_simulazioni),
                    tasso_privo_rischio=float(tasso_rf),
                    seme=int(seme))

            st.session_state["sim"] = sim
            st.session_state["universo_sim"] = universo_sim
            st.session_state["chiave"] = portfolio.portafogli_chiave(
                sim, MIN_TICKERS_FOR_MAX_RETURN)
            st.session_state["parametri"] = {
                "filtri": filtri, "ribilanciamento": ribilanciamento,
                "n_simulazioni": int(n_simulazioni), "seme": int(seme),
                "min_titoli": int(min_titoli), "max_titoli": int(max_titoli),
            }

    sim = st.session_state.get("sim")
    universo_sim = st.session_state.get("universo_sim")
    chiave = st.session_state.get("chiave", {})

    if sim is not None and not sim.portafogli.empty:
        st.success(
            f"{len(sim.portafogli):,} portafogli estratti in "
            f"{sim.secondi:.2f} secondi, su {len(sim.nomi)} titoli."
            .replace(",", "."))

        if universo_sim is not None and not universo_sim.scartati.empty:
            with st.expander(
                    f"{len(universo_sim.scartati)} titoli esclusi dalla "
                    "simulazione perche' troppo giovani", expanded=False):
                st.caption(
                    "Erano passati dallo screener ma la loro serie comincia "
                    "dopo la data limite. Un solo titolo troppo giovane "
                    "accorcia la storia comune di ogni portafoglio che lo "
                    "contiene, e rende i grafici incomparabili.")
                st.dataframe(universo_sim.scartati, width="stretch",
                             hide_index=True)

        st.divider()
        st.subheader("La frontiera efficiente")
        st.markdown(texts.FRONTIERA)
        campione = portfolio.campione_frontiera(sim, FRONTIER_MAX_POINTS)
        if len(campione) < len(sim.portafogli):
            st.caption(
                f"Nel grafico sono disegnati {len(campione):,} dei "
                f"{len(sim.portafogli):,} portafogli estratti: e' un campione "
                "casuale, e serve solo a non bloccare il browser. Tutte le "
                "statistiche e i tre portafogli notevoli usano l'insieme "
                "completo.".replace(",", "."))
        st.plotly_chart(
            charts.frontiera(campione, chiave,
                             f"Ribilanciamento: {REBALANCE_METHODS[ribilanciamento]}"),
            width="stretch")

        st.divider()
        st.subheader("I tre portafogli notevoli")
        st.markdown(texts.PORTAFOGLI_CHIAVE)

        colonne = st.columns(len(chiave) if chiave else 1)
        for colonna, (nome, dati) in zip(colonne, chiave.items()):
            with colonna:
                st.markdown(
                    f"<div style='border-left:4px solid "
                    f"{STRATEGY_COLORS.get(nome, COLORS['accent'])};"
                    f"padding-left:.7rem'><b>{nome}</b></div>",
                    unsafe_allow_html=True)
                st.metric("Rendimento annuo", pct(dati["rendimento"]))
                st.metric("Volatilita'", pct(dati["volatilita"]))
                st.metric("Sharpe", num(dati["sharpe"], 3))
                st.caption(f"{dati['num_tickers']} titoli: "
                           + ", ".join(dati["tickers"]))

        st.divider()
        st.subheader("I portafogli con lo Sharpe piu' alto")
        st.markdown(texts.TABELLA_TOP)
        quanti = st.slider("Quanti mostrarne", 5, 50, 15, 5)
        st.dataframe(
            portfolio.migliori(sim, quanti), width="stretch",
            hide_index=True,
            column_config={
                "Posizione": st.column_config.NumberColumn(width="small"),
                "Rendimento": st.column_config.NumberColumn(format="percent"),
                "Volatilita": st.column_config.NumberColumn(format="percent"),
                "Sharpe": st.column_config.NumberColumn(format="%.3f"),
            })


# --------------------------------------------------------------------------
# Scheda: Report
# --------------------------------------------------------------------------
with tab_report:
    sim = st.session_state.get("sim")
    universo_sim = st.session_state.get("universo_sim")
    chiave = st.session_state.get("chiave", {})

    if sim is None or universo_sim is None or not chiave:
        st.info("Il report si popola dopo aver estratto i portafogli. "
                "Vai alla scheda **Simulazione**.")
    else:
        parametri = st.session_state.get("parametri", {})
        metodo = parametri.get("ribilanciamento", ribilanciamento)

        # Ricostruire tre serie di portafoglio costa una decina di
        # millisecondi: metterle in cache costerebbe piu' righe di quante ne
        # faccia risparmiare, e una cache che si chiude sopra 'universo_sim'
        # rischierebbe di restituire serie vecchie dopo un nuovo screening.
        rendimenti = universo_sim.rendimenti
        serie = {
            nome: portfolio.serie_portafoglio(rendimenti, dati["tickers"], metodo)
            for nome, dati in chiave.items()
        }
        if universo_sim.benchmark:
            serie[NAME_BENCHMARK] = rendimenti[universo_sim.benchmark].dropna()
        serie = {nome: s for nome, s in serie.items() if not s.empty}

        curve = {nome: portfolio.equity(s) for nome, s in serie.items()}
        cadute = {nome: portfolio.drawdown(s) for nome, s in serie.items()}

        st.subheader("Le curve del capitale")
        st.markdown(texts.EQUITY)
        log_scale = st.checkbox("Scala logaritmica", value=True)
        st.plotly_chart(
            charts.equity(curve, f"Un euro investito - ribilanciamento: "
                                 f"{REBALANCE_METHODS[metodo]}", log_scale),
            width="stretch")

        st.subheader("Le misure di sintesi")
        righe = []
        for nome, s in serie.items():
            m = portfolio.misure(s, float(tasso_rf))
            if not m:
                continue
            righe.append({
                "Strategia": nome,
                "Dal": m["inizio"].date(), "Al": m["fine"].date(),
                "Anni": m["anni"], "CAGR": m["cagr"],
                "Volatilita": m["volatilita"], "Sharpe": m["sharpe"],
                "Drawdown max": m["max_drawdown"],
                "Rendimento totale": m["rendimento_totale"],
            })
        tabella_misure = pd.DataFrame(righe)
        st.dataframe(
            tabella_misure, width="stretch", hide_index=True,
            column_config={
                "Anni": st.column_config.NumberColumn(format="%.1f"),
                "CAGR": st.column_config.NumberColumn(format="percent"),
                "Volatilita": st.column_config.NumberColumn(format="percent"),
                "Sharpe": st.column_config.NumberColumn(format="%.3f"),
                "Drawdown max": st.column_config.NumberColumn(format="percent"),
                "Rendimento totale": st.column_config.NumberColumn(format="percent"),
            })

        st.divider()
        st.subheader("Quanto si e' stati sott'acqua")
        st.markdown(texts.DRAWDOWN)
        st.plotly_chart(charts.underwater(cadute), width="stretch")

        st.divider()
        st.subheader("Ogni possibile periodo pluriennale")
        st.markdown(texts.ROLLING)
        tabelle_rolling = {}
        schede = st.tabs(list(serie))
        for scheda, nome in zip(schede, serie):
            with scheda:
                tabella = portfolio.statistiche_rolling(serie[nome],
                                                        ROLLING_WINDOWS)
                tabelle_rolling[nome] = tabella
                st.dataframe(
                    tabella, width="stretch", hide_index=True,
                    column_config={
                        c: st.column_config.NumberColumn(format="percent")
                        for c in tabella.columns
                        if c not in ("Periodo", "Finestre")
                    })

        st.divider()
        st.subheader("Come si muovono insieme")
        st.markdown(texts.CORRELAZIONI)
        frequenti = portfolio.titoli_piu_frequenti(
            sim, HEATMAP_TOP_PORTFOLIOS, HEATMAP_MAX_TICKERS)
        matrice = portfolio.correlazioni(universo_sim.rendimenti, frequenti)
        st.plotly_chart(
            charts.correlazioni(
                matrice, f"I {len(matrice)} titoli piu' ricorrenti fra i "
                         f"{HEATMAP_TOP_PORTFOLIOS} portafogli migliori"),
            width="stretch")

        st.divider()
        st.subheader("Dentro un portafoglio")
        quale_pf = st.selectbox("Quale portafoglio aprire", list(chiave))
        st.plotly_chart(
            charts.contributo_titoli(universo_sim.rendimenti,
                                     chiave[quale_pf]["tickers"], quale_pf),
            width="stretch")

        st.divider()
        st.subheader("Esporta il report")
        st.caption("Un unico file HTML che si apre in qualunque browser e si "
                   "stampa in PDF con Stampa - Salva come PDF. Contiene i "
                   "grafici, le tabelle e i testi di lettura.")
        if st.button("Genera il report HTML", type="primary"):
            with st.spinner("Compongo il report..."):
                html = report.componi(
                    meta=meta, filtri=filtri, parametri=parametri,
                    riepilogo=riepilogo, chiave=chiave,
                    misure=tabella_misure,
                    migliori=portfolio.migliori(sim, 15),
                    rolling=tabelle_rolling,
                    figure={
                        "frontiera": charts.frontiera(
                            portfolio.campione_frontiera(sim, 6000), chiave),
                        "equity": charts.equity(curve, "", True),
                        "underwater": charts.underwater(cadute),
                        "correlazioni": charts.correlazioni(matrice),
                        "imbuto": charts.imbuto_filtri(riepilogo),
                    },
                )
            st.download_button(
                "Scarica il report", html.encode("utf-8"),
                file_name=f"report_screener_{dt.date.today().isoformat()}.html",
                mime="text/html", type="primary")


# --------------------------------------------------------------------------
# Scheda: Dati
# --------------------------------------------------------------------------
with tab_dati:
    st.subheader("L'archivio")
    st.markdown(texts.ARCHIVIO)

    stato = datastore.stato()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titoli nell'universo", num(meta.get("n_universo", len(metriche))))
    c2.metric("Con serie storiche", num(meta.get("n_con_prezzi", 0)))
    c3.metric("Costruito il", str(meta.get("costruito_il", ""))[:10] or "n.d.")
    c4.metric("Eta'", f"{eta:.0f} giorni" if eta is not None else "n.d.",
              delta=f"scade a {ARCHIVE_MAX_AGE_DAYS}",
              delta_color="inverse" if datastore.scaduto(meta) else "off")

    if datastore.scaduto(meta):
        avviso(f"<b>L'archivio ha piu' di {ARCHIVE_MAX_AGE_DAYS} giorni.</b> "
               "I risultati restano validi come esercizio, ma non contengono "
               "l'ultimo semestre. L'aggiornamento si lancia dalla scheda "
               "<i>Actions</i> del repository, workflow "
               "<i>aggiorna-archivio</i>, pulsante <i>Run workflow</i>.")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Come e' stato costruito**")
        dettagli = {
            "Universo": meta.get("universo_etichetta", "n.d."),
            "Borse": ", ".join(meta.get("borse", [])) or "n.d.",
            "Benchmark": meta.get("benchmark", BENCHMARK),
            "Anni di storico": meta.get("anni_storico", "n.d."),
            "Finestra": f"{meta.get('data_inizio', '?')} -> {meta.get('data_fine', '?')}",
            "Titoli misurati": num(meta.get("n_misurati", 0)),
            f"Con {meta.get('anni_storico', '?')} anni pieni":
                num(meta.get("n_con_storico_pieno", 0)),
            "Fette del pannello": meta.get("n_fette", "n.d."),
            "Durata del download": f"{meta.get('download_secondi', 0) / 60:.0f} minuti",
        }
        st.dataframe(
            pd.DataFrame({"Voce": list(dettagli), "Valore":
                          [str(v) for v in dettagli.values()]}),
            width="stretch", hide_index=True)

    with col_b:
        st.markdown("**Sul disco di questa macchina**")
        locali = {
            "Cartella": stato["cartella"],
            "Peso in cache": f"{stato['peso_mb']:.0f} MB",
            "Fette scaricate": f"{stato['fette_presenti']} su {stato['fette_attese']}",
        }
        st.dataframe(
            pd.DataFrame({"Voce": list(locali), "Valore": list(locali.values())}),
            width="stretch", hide_index=True)
        st.caption(
            "Le fette del pannello prezzi si scaricano solo quando servono: "
            "per lo screening bastano le misure, che pesano pochi megabyte.")
        if st.button("Svuota la cache locale"):
            datastore.svuota_cache()
            st.cache_data.clear()
            st.rerun()

        esiti = meta.get("esiti_download", {})
        if esiti:
            st.markdown("**Esito del download**")
            st.dataframe(
                pd.DataFrame({"Esito": list(esiti), "Titoli":
                              [num(v) for v in esiti.values()]}),
                width="stretch", hide_index=True)

    st.divider()
    st.subheader("Cerca un titolo")
    cerca = st.text_input("Ticker o nome", placeholder="AAPL, oppure Apple")
    if cerca:
        chiave_ricerca = cerca.strip().upper()
        trovati = risultato[risultato["ticker"].str.upper().str.contains(
            chiave_ricerca, na=False)]
        if trovati.empty and not anagrafica.empty:
            nomi = anagrafica[anagrafica["name"].str.upper().str.contains(
                chiave_ricerca, na=False)]["ticker"]
            trovati = risultato[risultato["ticker"].isin(nomi)]
        if trovati.empty:
            st.caption("Nessun titolo trovato con questo criterio.")
        else:
            mostra = trovati.head(40)
            if not anagrafica.empty:
                mostra = mostra.merge(anagrafica[["ticker", "name"]],
                                      on="ticker", how="left")
            st.dataframe(
                mostra[["ticker", "name", "esito", "anni_storico",
                        "prezzo_ultimo", f"dollar_volume_{LIQUIDITY_LOOKBACK}",
                        "cagr", "max_drawdown", "sharpe"]],
                width="stretch", hide_index=True,
                column_config={
                    "esito": st.column_config.TextColumn("Esito dei filtri"),
                    "anni_storico": st.column_config.NumberColumn(
                        "Anni", format="%.1f"),
                    "prezzo_ultimo": st.column_config.NumberColumn(
                        "Prezzo", format="$ %.2f"),
                    f"dollar_volume_{LIQUIDITY_LOOKBACK}":
                        st.column_config.NumberColumn("Volume $/gg",
                                                      format="compact"),
                    "cagr": st.column_config.NumberColumn(format="percent"),
                    "max_drawdown": st.column_config.NumberColumn(
                        "Drawdown", format="percent"),
                    "sharpe": st.column_config.NumberColumn(format="%.3f"),
                })


# --------------------------------------------------------------------------
# Scheda: Metodologia
# --------------------------------------------------------------------------
with tab_metodo:
    st.subheader("Come si usa")
    st.markdown(texts.COME_SI_USA)

    st.divider()
    st.subheader("Metodologia")
    st.markdown(texts.METODOLOGIA)

    st.divider()
    st.subheader("Glossario")
    for voce, spiegazione in texts.GLOSSARIO:
        with st.expander(voce):
            st.markdown(spiegazione)

    st.divider()
    st.subheader("Avvertenza")
    st.markdown(texts.DISCLAIMER)
