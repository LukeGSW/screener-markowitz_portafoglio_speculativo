"""
Esportazione del report in un unico file HTML.

E' l'erede della Cella 6 del notebook, con due differenze. La prima e' che i
grafici non sono immagini PNG incollate dentro la pagina ma **grafici veri**,
che si possono ingrandire e interrogare col mouse anche mesi dopo, senza
Python installato. La seconda e' che la pagina si compone in memoria e viene
consegnata come download, invece di essere scritta in una cartella che su
Colab sparisce alla chiusura della sessione.

Il file e' autonomo: dentro c'e' anche la libreria dei grafici, quindi si
apre in qualunque browser senza connessione. Pesa qualche megabyte, e va
bene cosi'.

I colori sono chiari, non scuri: questo documento e' fatto anche per essere
stampato, e uno sfondo scuro su carta e' uno spreco di toner e una fatica
per gli occhi.
"""

from __future__ import annotations

import datetime as dt
import html as html_mod
import re

import numpy as np
import pandas as pd

from . import texts
from .config import (APP_COURSE, APP_SUBTITLE, APP_TITLE, LIQUIDITY_LOOKBACK,
                     REBALANCE_METHODS)

# --------------------------------------------------------------------------
# Formattazione all'italiana
# --------------------------------------------------------------------------
def _it(valore: float, decimali: int = 2) -> str:
    s = f"{valore:,.{decimali}f}"
    return s.translate(str.maketrans({",": ".", ".": ","}))


def num(valore, decimali: int = 0) -> str:
    if valore is None or (isinstance(valore, float) and not np.isfinite(valore)):
        return "n.d."
    try:
        return _it(float(valore), decimali)
    except (TypeError, ValueError):
        return str(valore)


def pct(valore, decimali: int = 2) -> str:
    if valore is None or (isinstance(valore, float) and not np.isfinite(valore)):
        return "n.d."
    try:
        return f"{_it(float(valore) * 100.0, decimali)}%"
    except (TypeError, ValueError):
        return str(valore)


# --------------------------------------------------------------------------
# Un traduttore Markdown minimo
# --------------------------------------------------------------------------
# Non serve una libreria: i testi sono nostri, li conosciamo, e usano cinque
# costrutti in tutto. Meglio quaranta righe leggibili che una dipendenza in
# piu' da installare su Streamlit Cloud.
def markdown(testo: str) -> str:
    """Converte i testi didattici in HTML. Gestisce titoli, elenchi, enfasi."""
    righe = (testo or "").strip().split("\n")
    fuori: list[str] = []
    in_elenco = False
    in_codice = False
    paragrafo: list[str] = []

    def chiudi_paragrafo() -> None:
        nonlocal paragrafo
        if paragrafo:
            fuori.append(f"<p>{_inline(' '.join(paragrafo))}</p>")
            paragrafo = []

    def chiudi_elenco() -> None:
        nonlocal in_elenco
        if in_elenco:
            fuori.append("</ul>")
            in_elenco = False

    for riga in righe:
        nuda = riga.strip()

        if nuda.startswith("```"):
            chiudi_paragrafo()
            chiudi_elenco()
            fuori.append("</pre>" if in_codice else "<pre>")
            in_codice = not in_codice
            continue
        if in_codice:
            fuori.append(html_mod.escape(riga))
            continue

        if not nuda:
            chiudi_paragrafo()
            chiudi_elenco()
            continue

        livello = len(nuda) - len(nuda.lstrip("#"))
        if 0 < livello <= 6 and nuda[livello:livello + 1] == " ":
            chiudi_paragrafo()
            chiudi_elenco()
            grado = min(livello + 1, 6)   # ### del testo diventa <h4> nel report
            fuori.append(f"<h{grado}>{_inline(nuda[livello:].strip())}</h{grado}>")
            continue

        if nuda.startswith("- "):
            chiudi_paragrafo()
            if not in_elenco:
                fuori.append("<ul>")
                in_elenco = True
            fuori.append(f"<li>{_inline(nuda[2:].strip())}</li>")
            continue

        chiudi_elenco()
        paragrafo.append(nuda)

    chiudi_paragrafo()
    chiudi_elenco()
    if in_codice:
        fuori.append("</pre>")
    return "\n".join(fuori)


def _inline(testo: str) -> str:
    testo = html_mod.escape(testo)
    testo = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", testo)
    testo = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", testo)
    testo = re.sub(r"`([^`]+?)`", r"<code>\1</code>", testo)
    return testo


# --------------------------------------------------------------------------
# Grafici
# --------------------------------------------------------------------------
CHIARO = {
    "sfondo": "#FFFFFF",
    "griglia": "#E2E8F0",
    "testo": "#1E293B",
}


def _schiarisci(fig):
    """
    Riporta un grafico dal tema scuro dell'app a uno adatto alla carta.

    I colori delle serie restano quelli: teal, azzurro e viola si leggono
    bene anche su bianco, ed e' importante che il report e lo schermo
    parlino la stessa lingua visiva.
    """
    if fig is None:
        return None
    import plotly.graph_objects as go

    # Si lavora su una copia: la stessa figura viene mostrata a video nel
    # tema scuro, e non deve cambiare colore sotto i piedi dell'utente.
    nuova = go.Figure(fig.to_dict())
    nuova.update_layout(
        template="plotly_white",
        paper_bgcolor=CHIARO["sfondo"],
        plot_bgcolor=CHIARO["sfondo"],
        font=dict(color=CHIARO["testo"]),
        title=dict(font=dict(color=CHIARO["testo"])),
    )
    nuova.update_xaxes(gridcolor=CHIARO["griglia"], color=CHIARO["testo"])
    nuova.update_yaxes(gridcolor=CHIARO["griglia"], color=CHIARO["testo"])
    return nuova


def _grafico(fig, primo: bool = False) -> str:
    if fig is None:
        return "<p class='mancante'>Grafico non disponibile.</p>"
    import plotly.io as pio

    return pio.to_html(_schiarisci(fig), full_html=False,
                       include_plotlyjs=False, config={"displaylogo": False})


def _libreria() -> str:
    """La libreria dei grafici, incorporata una volta sola."""
    try:
        from plotly.offline import get_plotlyjs

        return f"<script>{get_plotlyjs()}</script>"
    except Exception:
        return ('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"'
                ' charset="utf-8"></script>')


# --------------------------------------------------------------------------
# Tabelle
# --------------------------------------------------------------------------
FORMATI = {
    "percent": pct,
    "numero": lambda v: num(v, 0),
    "decimale": lambda v: num(v, 3),
}


def _tabella(df: pd.DataFrame, formati: dict[str, str] | None = None,
             classe: str = "") -> str:
    """DataFrame -> tabella HTML, con i numeri gia' formattati all'italiana."""
    if df is None or df.empty:
        return "<p class='mancante'>Nessun dato.</p>"

    formati = formati or {}
    intestazione = "".join(f"<th>{html_mod.escape(str(c))}</th>" for c in df.columns)

    corpo = []
    for _, riga in df.iterrows():
        celle = []
        for colonna in df.columns:
            valore = riga[colonna]
            regola = formati.get(colonna)
            if regola in FORMATI:
                testo = FORMATI[regola](valore)
            elif isinstance(valore, float):
                testo = num(valore, 2)
            else:
                testo = "" if valore is None else str(valore)
            numerico = " class='num'" if regola or isinstance(
                valore, (int, float, np.number)) else ""
            celle.append(f"<td{numerico}>{html_mod.escape(testo)}</td>")
        corpo.append(f"<tr>{''.join(celle)}</tr>")

    return (f"<table class='{classe}'><thead><tr>{intestazione}</tr></thead>"
            f"<tbody>{''.join(corpo)}</tbody></table>")


def _schede_portafogli(chiave: dict) -> str:
    pezzi = []
    for nome, dati in chiave.items():
        pezzi.append(f"""
        <div class="scheda">
          <h4>{html_mod.escape(nome)}</h4>
          <div class="misure">
            <span><b>{pct(dati['rendimento'])}</b><small>rendimento annuo</small></span>
            <span><b>{pct(dati['volatilita'])}</b><small>volatilita'</small></span>
            <span><b>{num(dati['sharpe'], 3)}</b><small>Sharpe</small></span>
            <span><b>{dati['num_tickers']}</b><small>titoli</small></span>
          </div>
          <p class="titoli">{html_mod.escape(', '.join(dati['tickers']))}</p>
        </div>""")
    return "".join(pezzi)


# --------------------------------------------------------------------------
# Composizione
# --------------------------------------------------------------------------
STILE = """
:root { --testo:#1E293B; --tenue:#64748B; --bordo:#E2E8F0;
        --accento:#0D9488; --pannello:#F8FAFC; }
* { box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,
       'Helvetica Neue',Arial,sans-serif;
       color:var(--testo); background:#fff; margin:0; line-height:1.62;
       font-size:15px; }
.foglio { max-width:1180px; margin:0 auto; padding:2.4rem 1.6rem 4rem; }
header { border-bottom:3px solid var(--accento); padding-bottom:1.1rem;
         margin-bottom:2.2rem; }
header h1 { margin:0 0 .3rem; font-size:1.85rem; letter-spacing:-.01em; }
header .sotto { color:var(--tenue); font-size:.92rem; }
h2 { font-size:1.35rem; margin:2.8rem 0 .8rem; padding-bottom:.4rem;
     border-bottom:1px solid var(--bordo); }
h3 { font-size:1.1rem; margin:1.8rem 0 .5rem; }
h4 { font-size:1rem; margin:1.4rem 0 .4rem; color:var(--accento); }
p { margin:.7rem 0; }
ul { margin:.6rem 0 .9rem 1.2rem; padding-left:.6rem; }
li { margin:.3rem 0; }
code { background:var(--pannello); padding:.1rem .35rem; border-radius:3px;
       font-size:.9em; }
pre { background:var(--pannello); padding:.8rem 1rem; border-radius:6px;
      overflow-x:auto; font-size:.86rem; border:1px solid var(--bordo); }
table { border-collapse:collapse; width:100%; margin:1rem 0; font-size:.88rem; }
th { background:var(--pannello); text-align:left; padding:.55rem .7rem;
     border-bottom:2px solid var(--bordo); font-weight:600; }
td { padding:.45rem .7rem; border-bottom:1px solid var(--bordo); }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
tbody tr:hover { background:#FAFCFE; }
.riquadro { background:var(--pannello); border-left:3px solid var(--accento);
            border-radius:0 6px 6px 0; padding:.9rem 1.2rem; margin:1.2rem 0; }
.avviso { background:#FFFBEB; border-left:3px solid #D97706;
          border-radius:0 6px 6px 0; padding:.9rem 1.2rem; margin:1.2rem 0; }
.schede { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
          gap:1rem; margin:1.2rem 0; }
.scheda { border:1px solid var(--bordo); border-top:3px solid var(--accento);
          border-radius:6px; padding:1rem 1.1rem; }
.scheda h4 { margin:0 0 .7rem; color:var(--testo); }
.misure { display:flex; flex-wrap:wrap; gap:1.1rem; margin-bottom:.7rem; }
.misure span { display:flex; flex-direction:column; }
.misure b { font-size:1.15rem; font-variant-numeric:tabular-nums; }
.misure small { color:var(--tenue); font-size:.74rem; text-transform:uppercase;
                letter-spacing:.03em; }
.titoli { font-size:.82rem; color:var(--tenue); margin:0;
          font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace; }
.mancante { color:var(--tenue); font-style:italic; }
footer { margin-top:3.5rem; padding-top:1.2rem; border-top:1px solid var(--bordo);
         color:var(--tenue); font-size:.82rem; }
@media print {
  .foglio { max-width:none; padding:0; }
  h2 { page-break-after:avoid; }
  table, .scheda { page-break-inside:avoid; }
}
"""


def componi(meta: dict, filtri, parametri: dict, riepilogo: pd.DataFrame,
            chiave: dict, misure: pd.DataFrame, migliori: pd.DataFrame,
            rolling: dict[str, pd.DataFrame], figure: dict) -> str:
    """Mette insieme la pagina completa e la restituisce come stringa."""
    adesso = dt.datetime.now()
    metodo = parametri.get("ribilanciamento", "NONE")

    parametri_tabella = pd.DataFrame([
        ("Universo di partenza", meta.get("universo_etichetta", "n.d.")),
        ("Titoli esaminati", num(meta.get("n_universo", 0))),
        ("Archivio dati del", str(meta.get("costruito_il", ""))[:10]),
        ("Benchmark", meta.get("benchmark", "SPY.US")),
        ("Anni di storico richiesti", str(filtri.anni_storico)),
        ("Anni senza periodi in perdita", str(filtri.anni_perf_continua)),
        ("Drawdown massimo tollerato", pct(filtri.max_drawdown, 0)),
        ("Sharpe ratio minimo", num(filtri.min_sharpe, 2)),
        ("Volatilita' minima", pct(filtri.min_volatilita, 0)),
        ("Prezzo minimo", f"$ {num(filtri.prezzo_minimo, 2)}"),
        ("Prezzo massimo", f"$ {num(filtri.prezzo_massimo, 2)}"),
        (f"Volume in dollari minimo ({LIQUIDITY_LOOKBACK} gg)",
         f"$ {num(filtri.dollar_volume_minimo, 0)}"),
        ("Tasso privo di rischio", pct(filtri.tasso_privo_rischio, 2)),
        ("Titoli per portafoglio",
         f"da {parametri.get('min_titoli', '?')} a {parametri.get('max_titoli', '?')}"),
        ("Portafogli estratti", num(parametri.get("n_simulazioni", 0))),
        ("Ribilanciamento", REBALANCE_METHODS.get(metodo, metodo)),
        ("Seme casuale", str(parametri.get("seme", "n.d."))),
    ], columns=["Parametro", "Valore"])

    riepilogo_formattato = riepilogo.copy()
    riepilogo_formattato["Titoli"] = riepilogo["Titoli"].map(lambda v: num(v))
    riepilogo_formattato["Quota"] = riepilogo["Quota"].map(lambda v: pct(v, 2))

    sezioni_rolling = []
    for nome, tabella in rolling.items():
        sezioni_rolling.append(
            f"<h3>{html_mod.escape(nome)}</h3>"
            + _tabella(tabella, {c: "percent" for c in tabella.columns
                                 if c not in ("Periodo", "Finestre")})
        )

    corpo = f"""
<header>
  <h1>{APP_TITLE}</h1>
  <div class="sotto">{APP_SUBTITLE} &middot; {APP_COURSE}<br>
    Report generato il {adesso:%d/%m/%Y alle %H:%M}</div>
</header>

<div class="avviso">{markdown(texts.DISCLAIMER)}</div>

<h2>1. Che cosa e' stato chiesto</h2>
{_tabella(parametri_tabella)}

<h2>2. Dove si e' fermato l'universo</h2>
{markdown(texts.IMBUTO)}
{_tabella(riepilogo_formattato)}
{_grafico(figure.get('imbuto'))}

<h2>3. La frontiera efficiente</h2>
{markdown(texts.FRONTIERA)}
{_grafico(figure.get('frontiera'))}

<h2>4. I tre portafogli notevoli</h2>
{markdown(texts.PORTAFOGLI_CHIAVE)}
<div class="schede">{_schede_portafogli(chiave)}</div>

<h3>Le misure di sintesi</h3>
{_tabella(misure, {"CAGR": "percent", "Volatilita": "percent",
                   "Drawdown max": "percent",
                   "Rendimento totale": "percent", "Sharpe": "decimale"})}

<h3>I portafogli con lo Sharpe piu' alto</h3>
{markdown(texts.TABELLA_TOP)}
{_tabella(migliori, {"Rendimento": "percent", "Volatilita": "percent",
                     "Sharpe": "decimale"})}

<h2>5. Le curve del capitale</h2>
{markdown(texts.EQUITY)}
{_grafico(figure.get('equity'))}

<h2>6. Quanto si e' stati sott'acqua</h2>
{markdown(texts.DRAWDOWN)}
{_grafico(figure.get('underwater'))}

<h2>7. Ogni possibile periodo pluriennale</h2>
{markdown(texts.ROLLING)}
{''.join(sezioni_rolling)}

<h2>8. Come si muovono insieme</h2>
{markdown(texts.CORRELAZIONI)}
{_grafico(figure.get('correlazioni'))}

<h2>9. Metodologia</h2>
{markdown(texts.METODOLOGIA)}

<footer>
  {APP_TITLE} &middot; {APP_SUBTITLE}<br>
  Dati EOD Historical Data. Strumento didattico: i risultati sono simulazioni
  su dati storici e non costituiscono consulenza finanziaria.
</footer>
"""

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{APP_TITLE} - report del {adesso:%d/%m/%Y}</title>
<style>{STILE}</style>
{_libreria()}
</head>
<body><div class="foglio">{corpo}</div></body>
</html>"""
