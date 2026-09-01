"""
Grafici dello Screener (Plotly, tema scuro coerente con il Simulatore di
Portafogli del corso).

Tutte le funzioni ricevono dati gia' calcolati: qui non si fa alcuna
elaborazione finanziaria, solo rappresentazione. I grafici del notebook
erano immagini PNG prodotte da Matplotlib e incollate in una pagina HTML;
qui sono interattivi, e la differenza si sente soprattutto sulla frontiera
efficiente, dove passare il mouse su un punto dice finalmente di quale
portafoglio si tratta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import (COLORS, NAME_BENCHMARK, PLOTLY_TEMPLATE, REJECT_LABELS,
                     STRATEGY_COLORS)

SIMBOLI = {
    "Max Sharpe": "star",
    "Minima volatilita": "x",
    "Massimo rendimento": "cross",
}


def _base(fig: go.Figure, altezza: int = 560, titolo: str = "",
          legenda: bool = True) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=altezza,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=70 if titolo else 46, b=10),
        showlegend=legenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        # Separatori all'italiana: virgola per i decimali, punto per le migliaia.
        separators=",.",
    )
    if titolo:
        fig.update_layout(title=dict(text=titolo,
                                     font=dict(size=16, color=COLORS["text"])))
    # automargin: senza, i margini stretti tagliano i titoli degli assi e le
    # etichette lunghe delle barre orizzontali.
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False, automargin=True)
    return fig


def vuoto(messaggio: str = "Dati non disponibili") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=messaggio, showarrow=False,
                       font=dict(color=COLORS["muted"], size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, altezza=220, legenda=False)


# --------------------------------------------------------------------------
# La frontiera efficiente
# --------------------------------------------------------------------------
def frontiera(campione: pd.DataFrame, chiave: dict[str, dict],
              titolo: str = "") -> go.Figure:
    """
    La nuvola dei portafogli simulati nel piano rischio-rendimento.

    Ogni punto e' un portafoglio: sull'asse orizzontale quanto oscilla,
    sull'asse verticale quanto ha reso, e il colore dice il rapporto fra le
    due cose. Il bordo superiore sinistro della nuvola e' la frontiera
    efficiente: li' stanno i portafogli per cui non esiste un'alternativa
    che renda di piu' a parita' di rischio.
    """
    if campione is None or campione.empty:
        return vuoto("Nessun portafoglio simulato")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=campione["Volatility"], y=campione["Return"],
        mode="markers",
        marker=dict(
            size=4, opacity=0.55,
            color=campione["Sharpe_Ratio"],
            colorscale="Viridis",
            colorbar=dict(title=dict(text="Sharpe", side="right"),
                          thickness=12, len=0.75,
                          tickfont=dict(color=COLORS["muted"], size=10)),
        ),
        name="Portafogli simulati",
        customdata=np.stack([campione["Num_Tickers"],
                             campione["Sharpe_Ratio"]], axis=-1),
        hovertemplate=("Volatilita' %{x:.1%}<br>Rendimento %{y:.1%}"
                       "<br>%{customdata[0]} titoli"
                       "<br>Sharpe %{customdata[1]:.3f}<extra></extra>"),
    ))

    for nome, dati in chiave.items():
        fig.add_trace(go.Scatter(
            x=[dati["volatilita"]], y=[dati["rendimento"]],
            mode="markers", name=nome,
            marker=dict(size=18, symbol=SIMBOLI.get(nome, "circle"),
                        color=STRATEGY_COLORS.get(nome, COLORS["accent"]),
                        line=dict(width=1.5, color=COLORS["bg"])),
            hovertemplate=(f"<b>{nome}</b><br>Volatilita' %{{x:.1%}}"
                           "<br>Rendimento %{y:.1%}"
                           f"<br>{dati['num_tickers']} titoli"
                           f"<br>Sharpe {dati['sharpe']:.3f}<extra></extra>"),
        ))

    # Un decimale, non zero: la nuvola dei portafogli occupa spesso una
    # manciata di punti percentuali, e arrotondando all'intero le etichette
    # dell'asse si ripetono.
    fig.update_xaxes(title="Volatilita' annualizzata", tickformat=".1%")
    fig.update_yaxes(title="Rendimento annualizzato", tickformat=".1%")
    return _base(fig, altezza=620, titolo=titolo)


# --------------------------------------------------------------------------
# Curve del capitale
# --------------------------------------------------------------------------
def equity(curve: dict[str, pd.Series], titolo: str = "",
           scala_log: bool = False) -> go.Figure:
    """
    Quanto sarebbe diventato un euro investito all'inizio, per ciascuna strategia.

    Il benchmark e' tratteggiato: e' il termine di paragone, non una
    strategia in gara. Su vent'anni conviene quasi sempre la scala
    logaritmica, altrimenti il primo decennio si schiaccia sull'asse e
    sembra che non sia successo niente.
    """
    curve = {k: v for k, v in (curve or {}).items() if v is not None and not v.empty}
    if not curve:
        return vuoto("Nessuna serie da rappresentare")

    fig = go.Figure()
    for nome, serie in curve.items():
        benchmark = nome == NAME_BENCHMARK
        fig.add_trace(go.Scatter(
            x=serie.index, y=serie.to_numpy(),
            mode="lines", name=nome,
            line=dict(color=STRATEGY_COLORS.get(nome, COLORS["accent"]),
                      width=1.6 if benchmark else 2.1,
                      dash="dash" if benchmark else "solid"),
            hovertemplate=f"{nome}: %{{y:,.2f}}<extra></extra>",
        ))

    fig.update_xaxes(title="")
    fig.update_yaxes(title="Valore di 1 euro investito",
                     type="log" if scala_log else "linear",
                     tickformat=",.2f")
    fig.update_layout(hovermode="x unified")
    return _base(fig, altezza=560, titolo=titolo)


def underwater(serie_drawdown: dict[str, pd.Series], titolo: str = "") -> go.Figure:
    """
    Il grafico "sott'acqua": quanto si sta sotto il massimo precedente.

    E' il grafico che andrebbe guardato per primo. La curva del capitale
    dice dove si e' arrivati; questo dice che cosa si e' dovuto sopportare
    per arrivarci, ed e' l'unica delle due cose che si vive davvero.
    """
    serie_drawdown = {k: v for k, v in (serie_drawdown or {}).items()
                      if v is not None and not v.empty}
    if not serie_drawdown:
        return vuoto("Nessuna serie da rappresentare")

    fig = go.Figure()
    for nome, serie in serie_drawdown.items():
        colore = STRATEGY_COLORS.get(nome, COLORS["negative"])
        fig.add_trace(go.Scatter(
            x=serie.index, y=serie.to_numpy(),
            mode="lines", name=nome,
            line=dict(color=colore, width=1.5),
            fill="tozeroy",
            fillcolor=_trasparente(colore, 0.18),
            hovertemplate=f"{nome}: %{{y:.1%}}<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color=COLORS["muted"], width=1, dash="dot"))
    fig.update_yaxes(title="Distanza dal massimo", tickformat=".0%")
    fig.update_layout(hovermode="x unified")
    return _base(fig, altezza=420, titolo=titolo)


def _trasparente(colore: str, alfa: float) -> str:
    colore = colore.lstrip("#")
    r, g, b = (int(colore[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alfa})"


# --------------------------------------------------------------------------
# Diagnostica dello screener
# --------------------------------------------------------------------------
def imbuto_filtri(riepilogo: pd.DataFrame, titolo: str = "") -> go.Figure:
    """
    Dove si ferma l'universo: quanti titoli cadono su ciascun filtro.

    Va letto come una cascata. Il primo filtro lavora su tutti i titoli, il
    secondo solo sui superstiti del primo, e cosi' via. La barra piu' lunga
    e' il vincolo che sta davvero decidendo la selezione.
    """
    if riepilogo is None or riepilogo.empty:
        return vuoto("Nessun risultato di screening")

    df = riepilogo[riepilogo["Titoli"] > 0].copy()
    if df.empty:
        return vuoto("Nessun titolo nell'universo")
    df = df.iloc[::-1]

    passati = REJECT_LABELS["PASSATO"]
    colori = [COLORS["accent"] if e == passati else COLORS["grid"]
              for e in df["Esito"]]

    fig = go.Figure(go.Bar(
        x=df["Titoli"], y=df["Esito"], orientation="h",
        marker=dict(color=colori,
                    line=dict(color=COLORS["muted"], width=0.5)),
        text=[f"{n:,}".replace(",", ".") for n in df["Titoli"]],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
        hovertemplate="%{y}<br>%{x:,} titoli<extra></extra>",
    ))
    fig.update_xaxes(title="Numero di titoli")
    fig.update_yaxes(title="")
    altezza = max(320, 42 * len(df) + 120)
    return _base(fig, altezza=altezza, titolo=titolo, legenda=False)


def nuvola_universo(risultato: pd.DataFrame, massimo: int = 6000,
                    titolo: str = "") -> go.Figure:
    """
    L'intero universo nel piano rischio-rendimento, con i promossi in evidenza.

    Serve a rendersi conto di che cosa lo screener stia effettivamente
    scegliendo: quasi sempre un angolo molto piccolo e molto particolare
    della nuvola, e vedere quale angolo e' il modo piu' rapido per capire se
    i filtri stanno facendo quello che si crede.
    """
    if risultato is None or risultato.empty:
        return vuoto("Nessun dato")

    df = risultato.dropna(subset=["dev_std_ann", "cagr"]).copy()
    if df.empty:
        return vuoto("Nessun titolo misurabile")

    passati = df[df["esito"] == "PASSATO"]
    scartati = df[df["esito"] != "PASSATO"]
    if len(scartati) > massimo:
        scartati = scartati.sample(massimo, random_state=0)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=scartati["dev_std_ann"], y=scartati["cagr"],
        mode="markers", name="Scartati",
        marker=dict(size=3.5, color=COLORS["grid"], opacity=0.55),
        customdata=scartati[["ticker", "esito"]].to_numpy(),
        hovertemplate=("%{customdata[0]}<br>volatilita' %{x:.1%}"
                       "<br>CAGR %{y:.1%}<br>%{customdata[1]}<extra></extra>"),
    ))
    fig.add_trace(go.Scattergl(
        x=passati["dev_std_ann"], y=passati["cagr"],
        mode="markers", name="Promossi",
        marker=dict(size=7, color=COLORS["accent"],
                    line=dict(width=0.5, color=COLORS["bg"])),
        customdata=passati[["ticker"]].to_numpy(),
        hovertemplate=("<b>%{customdata[0]}</b><br>volatilita' %{x:.1%}"
                       "<br>CAGR %{y:.1%}<extra></extra>"),
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["muted"], width=1, dash="dot"))
    fig.update_xaxes(title="Volatilita' annualizzata (rendimenti giornalieri)",
                     tickformat=".0%")
    fig.update_yaxes(title="Crescita annua composta", tickformat=".0%")

    # Si taglia via l'un per cento estremo di ciascun asse: bastano tre
    # titoli sopravvissuti a un reverse split per schiacciare tutti gli
    # altri in una riga di pixel. Il taglio si applica solo se produce un
    # intervallo sensato - su un archivio piccolo i quantili degenerano.
    x_alto = float(df["dev_std_ann"].quantile(0.99))
    y_basso = float(df["cagr"].quantile(0.01))
    y_alto = float(df["cagr"].quantile(0.99))
    if np.isfinite(x_alto) and x_alto > 0:
        fig.update_xaxes(range=[0, x_alto * 1.03])
    if np.isfinite(y_basso) and np.isfinite(y_alto) and y_alto > y_basso:
        margine = (y_alto - y_basso) * 0.04
        fig.update_yaxes(range=[y_basso - margine, y_alto + margine])
    return _base(fig, altezza=560, titolo=titolo)


def sensibilita(curva: pd.DataFrame, etichetta: str, valore_attuale: float,
                percentuale: bool = False, titolo: str = "") -> go.Figure:
    """Quanti titoli sopravvivono al variare di una sola soglia."""
    if curva is None or curva.empty:
        return vuoto("Nessun dato")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curva["valore"], y=curva["promossi"],
        mode="lines+markers", name="Titoli promossi",
        line=dict(color=COLORS["accent"], width=2.2),
        marker=dict(size=6),
        hovertemplate="%{x}<br>%{y:,} titoli<extra></extra>",
    ))
    fig.add_vline(x=valore_attuale,
                  line=dict(color=COLORS["warning"], width=1.5, dash="dash"),
                  annotation_text="impostazione attuale",
                  annotation_font=dict(color=COLORS["warning"], size=11))
    fig.update_xaxes(title=etichetta, tickformat=".0%" if percentuale else None)
    fig.update_yaxes(title="Titoli che superano tutti i filtri")
    return _base(fig, altezza=360, titolo=titolo, legenda=False)


# --------------------------------------------------------------------------
# Correlazioni
# --------------------------------------------------------------------------
def correlazioni(matrice: pd.DataFrame, titolo: str = "") -> go.Figure:
    """
    Quanto si muovono insieme i titoli piu' ricorrenti nei portafogli migliori.

    Si cercano valori bassi o negativi: se tutto e' rosso acceso, la
    diversificazione e' apparente e il portafoglio ha in realta' una sola
    scommessa dentro, ripetuta cinque volte.
    """
    if matrice is None or matrice.empty or matrice.shape[0] < 2:
        return vuoto("Servono almeno due titoli per una matrice di correlazione")

    testo = [[f"{v:.2f}" for v in riga] for riga in matrice.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=matrice.to_numpy(),
        x=list(matrice.columns), y=list(matrice.index),
        zmin=-1, zmax=1, colorscale="RdBu_r",
        text=testo, texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(thickness=12, len=0.8,
                      tickfont=dict(color=COLORS["muted"], size=10)),
        hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
    ))
    lato = max(420, 34 * len(matrice) + 160)
    fig.update_yaxes(autorange="reversed")
    return _base(fig, altezza=lato, titolo=titolo, legenda=False)


# --------------------------------------------------------------------------
# Composizione di un portafoglio
# --------------------------------------------------------------------------
def contributo_titoli(rendimenti: pd.DataFrame, tickers, nome: str) -> go.Figure:
    """
    Crescita di ciascun titolo del portafoglio, presi uno per uno.

    Mostra la dispersione che si nasconde dentro una media: un portafoglio
    equipesato con un buon rendimento complessivo puo' contenere un titolo
    che ha fatto dieci volte il capitale e tre che l'hanno dimezzato.
    """
    presenti = [t for t in tickers if t in rendimenti.columns]
    if not presenti:
        return vuoto("Nessun titolo disponibile")

    sotto = rendimenti[presenti].dropna()
    if sotto.empty:
        return vuoto("Nessun dato in comune fra i titoli")

    curve = (1.0 + sotto).cumprod()
    finali = curve.iloc[-1].sort_values(ascending=False)

    fig = go.Figure()
    for i, t in enumerate(finali.index):
        fig.add_trace(go.Scatter(
            x=curve.index, y=curve[t].to_numpy(), mode="lines", name=t,
            line=dict(width=1.4), opacity=0.9,
            hovertemplate=f"{t}: %{{y:,.2f}}<extra></extra>",
        ))
    fig.update_yaxes(title="Valore di 1 euro investito", type="log",
                     tickformat=",.2f")
    return _base(fig, altezza=460,
                 titolo=f"I singoli titoli di '{nome}', uno per uno")
