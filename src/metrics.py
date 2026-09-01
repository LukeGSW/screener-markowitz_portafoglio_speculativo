"""
Misure di un singolo titolo: il nucleo dello screener.

Qui sta la ragione per cui l'applicazione e' veloce dove il notebook era
lento. Il notebook, ogni volta che si cambiava una soglia, rileggeva da capo
ventitremila file CSV e ricalcolava tutto. Ma le misure di un titolo - da
quanto e' quotato, quanto scambia, quanto ha perso nel peggior momento, che
Sharpe ha avuto - NON DIPENDONO DALLE SOGLIE. Dipendono solo dai prezzi.

Quindi si calcolano una volta sola, quando si costruisce l'archivio, e si
mettono in tabella: una riga per titolo, una colonna per misura. Da quel
momento filtrare l'universo intero e' un confronto fra due colonne di
NumPy - qualche millisecondo - invece di una rilettura di tutto il disco.

Le formule sono quelle del notebook, riga per riga. Dove ci si discosta e'
segnalato nel commento, ed e' sempre nella direzione della prudenza: il
notebook, per come erano scritti i confronti, lasciava passare un titolo le
cui misure risultassero NaN (perche' "NaN < soglia" e' falso in NumPy).
Qui un titolo che non si riesce a misurare viene scartato, non promosso.
"""

from __future__ import annotations

import numpy as np

from .config import (DOLLARVOL_WINDOWS, LIQUIDITY_LOOKBACK, MAX_ROLLING_YEARS,
                     MIN_OBSERVATIONS, MONTHS_PER_YEAR, TRADING_DAYS_PER_YEAR)

# --------------------------------------------------------------------------
# Le quattro funzioni del notebook, invariate nella sostanza
# --------------------------------------------------------------------------


def max_drawdown(prezzi: np.ndarray) -> float:
    """
    Perdita massima dal massimo precedente, in frazione (es. -0,42).

    Si scorre la serie tenendo memoria del massimo raggiunto fino a quel
    punto; il drawdown del giorno e' quanto si sta sotto quel massimo. Il
    valore restituito e' il piu' profondo di tutta la storia.
    """
    if prezzi.size == 0:
        return np.nan
    massimo_corrente = np.maximum.accumulate(prezzi)
    massimo_corrente = np.where(massimo_corrente == 0, 1.0, massimo_corrente)
    return float((prezzi / massimo_corrente - 1.0).min())


def rendimenti_giornalieri(prezzi: np.ndarray) -> np.ndarray:
    """
    Rendimenti semplici giorno su giorno.

    Si saltano i giorni in cui il prezzo precedente non e' positivo: e' la
    stessa maschera del notebook, e serve a non dividere per zero sui titoli
    con quotazioni sporche.
    """
    if prezzi.size < 2:
        return np.empty(0, dtype=np.float64)
    precedenti = prezzi[:-1]
    validi = precedenti > 0
    if not validi.any():
        return np.empty(0, dtype=np.float64)
    return (prezzi[1:][validi] / precedenti[validi]) - 1.0


def volatilita_mensile_annualizzata(prezzi: np.ndarray,
                                    date: np.ndarray) -> float:
    """
    Volatilita' annualizzata calcolata sui rendimenti MENSILI.

    Non e' la volatilita' giornaliera moltiplicata per la radice di 252: si
    prende l'ultimo prezzo di ogni mese, se ne calcola la variazione, e si
    annualizza con la radice di 12. E' una misura piu' ruvida ma anche meno
    nervosa, e serve nel notebook come filtro di "vitalita'": un titolo che
    non si muove quasi mai non e' un candidato interessante per un
    portafoglio speculativo.

    Restituisce -1,0 quando i mesi disponibili sono meno di dodici: e' il
    valore sentinella del notebook, che fa fallire il filtro.
    """
    if prezzi.size == 0:
        return -1.0
    mesi = date.astype("datetime64[M]")
    fine_mese = np.empty(mesi.size, dtype=bool)
    fine_mese[-1] = True
    if mesi.size > 1:
        fine_mese[:-1] = mesi[:-1] != mesi[1:]

    prezzi_mensili = prezzi[fine_mese].astype(np.float64)
    if prezzi_mensili.size < MONTHS_PER_YEAR:
        return -1.0

    precedenti = prezzi_mensili[:-1]
    validi = precedenti != 0
    if not validi.any():
        return -1.0
    variazioni = (prezzi_mensili[1:][validi] / precedenti[validi]) - 1.0
    return float(np.std(variazioni) * np.sqrt(MONTHS_PER_YEAR))


def minimo_rendimento_rolling(prezzi: np.ndarray, anni: int) -> float:
    """
    Il peggiore fra tutti i rendimenti su finestra di 'anni' anni.

    E' il filtro della "performance continua": si confronta ogni giorno con
    il giorno di N anni prima e si guarda il caso peggiore. Se anche il
    peggiore e' positivo, vuol dire che nella storia del titolo non e' mai
    esistito un periodo di quella lunghezza chiuso in perdita.

    Restituisce NaN quando la storia non basta a formare nemmeno una
    finestra: in quel caso il filtro deve fallire, e infatti ogni confronto
    con NaN e' falso.
    """
    giorni = int(anni * TRADING_DAYS_PER_YEAR)
    if giorni <= 0 or prezzi.size <= giorni:
        return np.nan
    inizio = prezzi[:-giorni]
    fine = prezzi[giorni:]
    validi = inizio > 0
    if not validi.any():
        return np.nan
    return float(((fine[validi] / inizio[validi]) - 1.0).min())


# --------------------------------------------------------------------------
# La riga di tabella di un titolo
# --------------------------------------------------------------------------
# Ordine e tipo delle colonne prodotte da 'misura'. Serve a costruire il
# DataFrame in modo esplicito, senza affidarsi all'ordine dei dizionari.
COLONNE = {
    "ticker": "string",
    "n_oss": "int32",
    "data_inizio": "datetime64[ns]",
    "data_fine": "datetime64[ns]",
    "anni_storico": "float32",
    "prezzo_ultimo": "float32",
    "prezzo_primo": "float32",
    "max_drawdown": "float32",
    "media_ann": "float32",
    "dev_std_ann": "float32",
    "n_rendimenti": "int32",
    "volatilita_mensile": "float32",
    "cagr": "float32",
    "rendimento_totale": "float32",
}
for _f in DOLLARVOL_WINDOWS:
    COLONNE[f"dollar_volume_{_f}"] = "float64"
for _a in range(1, MAX_ROLLING_YEARS + 1):
    COLONNE[f"min_roll_{_a}a"] = "float32"


def misura(ticker: str, date: np.ndarray, prezzi: np.ndarray,
           volumi: np.ndarray | None = None) -> dict | None:
    """
    Calcola tutte le misure di un titolo a partire dalle sue serie.

    Riceve array gia' ordinati per data, senza duplicati e con prezzi
    positivi (ci pensa il downloader). Restituisce None se il titolo non ha
    nemmeno un anno di quotazioni, che e' il controllo con cui il notebook
    apriva la lavorazione di ogni file.
    """
    n = int(prezzi.size)
    if n < MIN_OBSERVATIONS:
        return None

    prezzi64 = prezzi.astype(np.float64)
    giorni_totali = (date[-1] - date[0]).astype("timedelta64[D]").astype(int)

    riga: dict = {
        "ticker": ticker,
        "n_oss": n,
        "data_inizio": date[0],
        "data_fine": date[-1],
        "anni_storico": giorni_totali / 365.25,
        "prezzo_ultimo": float(prezzi64[-1]),
        "prezzo_primo": float(prezzi64[0]),
        "max_drawdown": max_drawdown(prezzi64),
    }

    # --- Rendimenti, media e dispersione ---------------------------------
    # Si conservano media e deviazione standard separate invece dello Sharpe
    # gia' fatto: lo Sharpe dipende dal tasso privo di rischio, che e' un
    # parametro dello studente e va poter cambiare senza ricostruire nulla.
    rendimenti = rendimenti_giornalieri(prezzi64)
    riga["n_rendimenti"] = int(rendimenti.size)
    if rendimenti.size:
        riga["media_ann"] = float(np.mean(rendimenti) * TRADING_DAYS_PER_YEAR)
        riga["dev_std_ann"] = float(
            np.std(rendimenti) * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        riga["media_ann"] = np.nan
        riga["dev_std_ann"] = 0.0

    riga["volatilita_mensile"] = volatilita_mensile_annualizzata(prezzi64, date)

    # --- Crescita complessiva --------------------------------------------
    anni = giorni_totali / 365.25
    totale = prezzi64[-1] / prezzi64[0] - 1.0
    riga["rendimento_totale"] = float(totale)
    riga["cagr"] = float((1.0 + totale) ** (1.0 / anni) - 1.0) if anni > 0 else np.nan

    # --- Liquidita' -------------------------------------------------------
    # Volume in dollari = prezzo x pezzi scambiati, mediato sugli ultimi
    # giorni. E' la misura che separa un titolo su cui si puo' davvero
    # entrare e uscire da uno che esiste solo sulla carta.
    for finestra in DOLLARVOL_WINDOWS:
        chiave = f"dollar_volume_{finestra}"
        if volumi is None or n <= finestra:
            # Il notebook, quando la storia era piu' corta della finestra,
            # semplicemente non applicava il filtro. NaN ottiene lo stesso
            # effetto: ogni confronto con NaN e' falso.
            riga[chiave] = np.nan
        else:
            v = np.nan_to_num(volumi[-finestra:].astype(np.float64), nan=0.0)
            riga[chiave] = float(np.mean(prezzi64[-finestra:] * v))

    # --- Performance continua, tutte le finestre da 1 a 10 anni ----------
    # Calcolarle tutte adesso costa qualche microsecondo e permette allo
    # studente di spostare il cursore degli anni senza alcuna attesa.
    for a in range(1, MAX_ROLLING_YEARS + 1):
        riga[f"min_roll_{a}a"] = minimo_rendimento_rolling(prezzi64, a)

    return riga


def riga_vuota(ticker: str) -> dict:
    """Riga segnaposto per un titolo scaricato ma non misurabile."""
    riga = {c: np.nan for c in COLONNE}
    riga["ticker"] = ticker
    riga["n_oss"] = 0
    riga["n_rendimenti"] = 0
    riga["data_inizio"] = np.datetime64("NaT")
    riga["data_fine"] = np.datetime64("NaT")
    return riga
