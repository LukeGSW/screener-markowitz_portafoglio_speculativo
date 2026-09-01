"""
Costruzione e misura dei portafogli.

Il metodo e' quello del notebook: si estraggono a sorte decine di migliaia di
combinazioni di titoli fra quelli sopravvissuti allo screener, si assegna a
ciascun titolo lo stesso peso, e si guarda dove finiscono nel piano
rischio-rendimento. I portafogli che stanno sul bordo superiore sinistro
della nuvola sono la frontiera efficiente.

La differenza sta nel come. Il notebook girava un ciclo Python di
cinquantamila iterazioni, ciascuna con la sua estrazione e il suo piccolo
prodotto matriciale: un minuto abbondante. Qui le cinquantamila estrazioni
sono un'unica operazione NumPy, fatta a blocchi per non occupare memoria:
un paio di secondi. La matematica e' identica - stessa media, stessa
varianza, stesso Sharpe - cambia solo che si fa tutta in una volta.

Lo stesso vale per le statistiche rolling, che nel notebook usavano
'.rolling().apply()' con una funzione Python chiamata una volta per ogni
giorno di ogni finestra di ogni portafoglio, ed erano di gran lunga la parte
piu' lenta della visualizzazione.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (HISTORY_TOLERANCE_DAYS, MIN_TICKERS_FOR_MAX_RETURN,
                     NAME_MAX_RET, NAME_MAX_SHARPE, NAME_MIN_VOL,
                     TRADING_DAYS_PER_YEAR)

# Quanti numeri casuali tenere in memoria contemporaneamente durante
# l'estrazione. Quattro milioni di float sono trentadue megabyte: abbastanza
# per essere efficienti, abbastanza poco per girare ovunque.
BLOCCO_CASUALE = 4_000_000

# Idem per le sottomatrici di covarianza, che sono blocco x k x k.
BLOCCO_COVARIANZE = 2_000_000

# Quante finestre alla volta nel calcolo dei drawdown rolling.
BLOCCO_FINESTRE = 512


# --------------------------------------------------------------------------
# Preparazione dei rendimenti
# --------------------------------------------------------------------------
@dataclass
class Universo:
    """Il materiale di partenza della simulazione, gia' ripulito."""

    rendimenti: pd.DataFrame              # date x ticker, benchmark incluso
    tickers: list[str]                    # titoli utilizzabili (benchmark escluso)
    benchmark: str | None
    scartati: pd.DataFrame                # titoli esclusi per storico corto
    data_limite: pd.Timestamp | None


def prepara_universo(prezzi: pd.DataFrame, benchmark: str | None,
                     anni_storico: int) -> Universo:
    """
    Trasforma i prezzi in rendimenti ed espelle i titoli troppo giovani.

    E' la "caccia all'intruso" del notebook, e serve a un problema concreto:
    un titolo quotato da tre anni, se entra nel calcolo, accorcia di
    diciassette anni la storia comune di qualunque portafoglio lo contenga,
    e i grafici diventano incomparabili. Lo screener dovrebbe averli gia'
    esclusi tutti, ma il controllo si ripete qui perche' costa nulla e
    perche' un solo intruso rovina l'intera schermata.

    La tolleranza di sessanta giorni assorbe il fatto che i calendari di
    borsa non partono mai lo stesso identico giorno.
    """
    if prezzi.empty:
        return Universo(pd.DataFrame(), [], benchmark, pd.DataFrame(), None)

    prezzi = prezzi.sort_index()
    # Rendimenti semplici senza alcun riempimento dei buchi: un giorno senza
    # prezzo resta un giorno senza rendimento, e si vedra' piu' avanti come
    # trattarlo (azzerato per le matrici, eliminato per le serie storiche).
    rendimenti = prezzi / prezzi.shift(1) - 1.0

    fine = rendimenti.index.max()
    limite = fine - pd.Timedelta(days=anni_storico * 365 - HISTORY_TOLERANCE_DAYS)

    candidati = [c for c in rendimenti.columns if c != benchmark]
    buoni, cattivi = [], []
    for t in candidati:
        primo = rendimenti[t].first_valid_index()
        if primo is None:
            cattivi.append({"ticker": t, "inizio": pd.NaT, "anni": 0.0})
            continue
        if primo > limite:
            cattivi.append({
                "ticker": t,
                "inizio": primo,
                "anni": (fine - primo).days / 365.25,
            })
        else:
            buoni.append(t)

    colonne = buoni + ([benchmark] if benchmark in rendimenti.columns else [])
    return Universo(
        rendimenti=rendimenti[colonne],
        tickers=buoni,
        benchmark=benchmark if benchmark in rendimenti.columns else None,
        scartati=pd.DataFrame(cattivi, columns=["ticker", "inizio", "anni"]),
        data_limite=limite,
    )


def matrici(universo: Universo) -> tuple[np.ndarray, np.ndarray]:
    """
    Rendimento atteso e matrice di covarianza, entrambi annualizzati.

    I buchi vengono azzerati prima del calcolo, come nel notebook: e' una
    scelta discutibile in generale, ma su titoli con vent'anni di storia i
    buchi sono pochissimi e azzerarli e' meno distorsivo che eliminare
    l'intera giornata per tutti.
    """
    if not universo.tickers:
        return np.empty(0), np.empty((0, 0))
    dati = universo.rendimenti[universo.tickers].fillna(0.0)
    mu = dati.mean().to_numpy(dtype=np.float64) * TRADING_DAYS_PER_YEAR
    cov = dati.cov().to_numpy(dtype=np.float64) * TRADING_DAYS_PER_YEAR
    return mu, cov


# --------------------------------------------------------------------------
# Monte Carlo vettoriale
# --------------------------------------------------------------------------
@dataclass
class Simulazione:
    """Esito di una campagna Monte Carlo."""

    portafogli: pd.DataFrame              # Num_Tickers, Return, Volatility, Sharpe_Ratio
    indici: np.ndarray                    # (n_sim, k_max) posizioni, -1 = vuoto
    nomi: list[str]                       # nomi dei titoli, per posizione
    secondi: float = 0.0

    def tickers(self, riga: int) -> tuple[str, ...]:
        """I titoli del portafoglio simulato in posizione 'riga'."""
        posizioni = self.indici[riga]
        return tuple(sorted(self.nomi[p] for p in posizioni if p >= 0))


def simula(mu: np.ndarray, cov: np.ndarray, nomi: list[str],
           k_min: int, k_max: int, n_simulazioni: int,
           tasso_privo_rischio: float = 0.0,
           seme: int | None = None,
           on_progress=None) -> Simulazione:
    """
    Estrae n portafogli equipesati e ne calcola rendimento, rischio e Sharpe.

    Per un portafoglio con pesi tutti uguali a 1/k:

        rendimento atteso = media dei rendimenti attesi dei k titoli
        varianza          = somma di tutta la sottomatrice di covarianza / k^2

    Sono le stesse due formule del notebook. La differenza e' che qui la
    sottomatrice viene estratta per migliaia di portafogli in un colpo solo,
    con l'indicizzazione avanzata di NumPy, invece che uno alla volta.
    """
    avvio = dt.datetime.now()
    n = len(nomi)
    if n == 0 or n_simulazioni <= 0:
        vuoto = pd.DataFrame(columns=["Num_Tickers", "Return", "Volatility",
                                      "Sharpe_Ratio"])
        return Simulazione(vuoto, np.empty((0, 0), dtype=np.int32), nomi)

    k_min = max(1, min(int(k_min), n))
    k_max = max(k_min, min(int(k_max), n))
    n_simulazioni = int(n_simulazioni)

    rng = np.random.default_rng(seme)
    # Numero di titoli di ciascun portafoglio: uniforme fra minimo e massimo,
    # estremi inclusi, come il random.randint del notebook.
    ks = rng.integers(k_min, k_max + 1, size=n_simulazioni)

    indici = np.full((n_simulazioni, k_max), -1, dtype=np.int32)
    rendimento = np.empty(n_simulazioni, dtype=np.float64)
    varianza = np.empty(n_simulazioni, dtype=np.float64)

    fatti = 0
    for k in np.unique(ks):
        k = int(k)
        posizioni = np.flatnonzero(ks == k)

        # Quante estrazioni alla volta. Due vincoli, e conta il piu' stretto:
        # la matrice dei sorteggi e' grande blocco x numero di titoli, e le
        # sottomatrici di covarianza sono blocco x k x k. Con portafogli da
        # quaranta titoli il secondo vincolo diventa quello che decide.
        blocco = max(256, min(20_000,
                              BLOCCO_CASUALE // max(n, 1),
                              BLOCCO_COVARIANZE // max(k * k, 1)))

        for inizio in range(0, posizioni.size, blocco):
            righe = posizioni[inizio:inizio + blocco]
            b = righe.size

            # Estrazione di k indici distinti su n, per b portafogli: si
            # sorteggia un numero per ciascun titolo e si prendono i k piu'
            # piccoli. E' il modo piu' rapido di ottenere sottoinsiemi
            # uniformi senza reinserimento in blocco.
            sorteggio = rng.random((b, n))
            if k < n:
                scelti = np.argpartition(sorteggio, k - 1, axis=1)[:, :k]
            else:
                scelti = np.tile(np.arange(n), (b, 1))

            indici[righe, :k] = scelti.astype(np.int32)
            rendimento[righe] = mu[scelti].mean(axis=1)
            # Somma dell'intera sottomatrice di covarianza, diviso k al quadrato.
            sotto = cov[scelti[:, :, None], scelti[:, None, :]]
            varianza[righe] = sotto.sum(axis=(1, 2)) / float(k * k)

            fatti += b
            if on_progress is not None:
                on_progress(fatti, n_simulazioni)

    volatilita = np.sqrt(np.maximum(varianza, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(volatilita > 0,
                          (rendimento - tasso_privo_rischio) / volatilita, 0.0)

    portafogli = pd.DataFrame({
        "Num_Tickers": ks.astype(np.int16),
        "Return": rendimento,
        "Volatility": volatilita,
        "Sharpe_Ratio": sharpe,
    })
    durata = (dt.datetime.now() - avvio).total_seconds()
    return Simulazione(portafogli, indici, list(nomi), durata)


def portafogli_chiave(sim: Simulazione,
                      min_titoli_max_rendimento: int = MIN_TICKERS_FOR_MAX_RETURN
                      ) -> dict[str, dict]:
    """
    I tre portafogli notevoli: massimo Sharpe, minima volatilita', massimo
    rendimento a parita' di diversificazione minima.

    Il vincolo sul numero minimo di titoli per il portafoglio di massimo
    rendimento non e' un dettaglio: senza, vincerebbe sistematicamente il
    portafoglio piu' concentrato, che e' un modo elaborato per dire "il
    titolo che e' salito di piu'".
    """
    df = sim.portafogli
    if df.empty:
        return {}

    scelte: dict[str, dict] = {}

    def aggiungi(nome: str, riga: int) -> None:
        scelte[nome] = {
            "riga": int(riga),
            "tickers": sim.tickers(riga),
            "num_tickers": int(df.at[riga, "Num_Tickers"]),
            "rendimento": float(df.at[riga, "Return"]),
            "volatilita": float(df.at[riga, "Volatility"]),
            "sharpe": float(df.at[riga, "Sharpe_Ratio"]),
        }

    aggiungi(NAME_MAX_SHARPE, df["Sharpe_Ratio"].idxmax())
    aggiungi(NAME_MIN_VOL, df["Volatility"].idxmin())

    ammessi = df.index[df["Num_Tickers"] >= min_titoli_max_rendimento]
    if len(ammessi):
        aggiungi(NAME_MAX_RET, df.loc[ammessi, "Return"].idxmax())

    return scelte


def migliori(sim: Simulazione, quanti: int = 15,
             per: str = "Sharpe_Ratio") -> pd.DataFrame:
    """Tabella dei portafogli migliori, con i nomi dei titoli espansi."""
    df = sim.portafogli
    if df.empty:
        return pd.DataFrame()
    top = df.nlargest(min(quanti, len(df)), per)
    righe = []
    for posto, (riga, dati) in enumerate(top.iterrows(), start=1):
        righe.append({
            "Posizione": posto,
            "Titoli": ", ".join(sim.tickers(riga)),
            "N. titoli": int(dati["Num_Tickers"]),
            "Rendimento": float(dati["Return"]),
            "Volatilita": float(dati["Volatility"]),
            "Sharpe": float(dati["Sharpe_Ratio"]),
        })
    return pd.DataFrame(righe)


def campione_frontiera(sim: Simulazione, massimo: int,
                       seme: int | None = 0) -> pd.DataFrame:
    """
    Sottoinsieme dei portafogli da disegnare.

    Cinquantamila punti in un grafico interattivo bloccano il browser. Si
    disegna un campione casuale, ma i portafogli notevoli vengono aggiunti a
    parte e restano sempre visibili: la nuvola serve a dare la forma, non a
    essere letta punto per punto.
    """
    df = sim.portafogli
    if df.empty or len(df) <= massimo:
        return df
    rng = np.random.default_rng(seme)
    scelte = rng.choice(len(df), size=massimo, replace=False)
    return df.iloc[np.sort(scelte)]


# --------------------------------------------------------------------------
# Serie storica di un portafoglio
# --------------------------------------------------------------------------
def _confini_ribilanciamento(indice: pd.DatetimeIndex, metodo: str) -> np.ndarray:
    """
    Posizioni in cui il portafoglio viene riportato ai pesi obiettivo.

    Si ribilancia sull'ULTIMO GIORNO DI BORSA del periodo, non sulla data di
    calendario: il 31 dicembre 2022 era un sabato, e un ribilanciamento
    fissato a quella data semplicemente non sarebbe avvenuto. Il notebook
    confrontava date di calendario con l'indice dei prezzi e saltava in
    silenzio i ribilanciamenti caduti nel fine settimana.
    """
    if metodo in ("NONE", "BUYHOLD") or len(indice) == 0:
        return np.empty(0, dtype=np.int64)

    anno = indice.year.to_numpy()
    mese = indice.month.to_numpy()
    if metodo == "QUARTERLY":
        periodo = anno * 4 + (mese - 1) // 3
    elif metodo == "SEMIANNUALLY":
        periodo = anno * 2 + (mese - 1) // 6
    elif metodo == "ANNUALLY":
        periodo = anno
    else:
        return np.empty(0, dtype=np.int64)

    ultimo = np.empty(periodo.size, dtype=bool)
    ultimo[-1] = False          # l'ultimo giorno della serie non serve
    ultimo[:-1] = periodo[:-1] != periodo[1:]
    return np.flatnonzero(ultimo)


def serie_portafoglio(rendimenti: pd.DataFrame, tickers, metodo: str = "NONE"
                      ) -> pd.Series:
    """
    Rendimenti giornalieri di un portafoglio equipesato.

    I cinque metodi disponibili sono davvero diversi fra loro:

    NONE      i pesi restano uguali OGNI GIORNO. Suona come "non faccio
              niente", ma e' l'opposto: equivale a ribilanciare tutte le
              sere. E' il comportamento del notebook, mantenuto perche' e'
              quello con cui sono stati prodotti i risultati del corso.
    BUYHOLD   si comprano quote uguali il primo giorno e non si tocca piu'
              nulla. I pesi derivano: dopo vent'anni il titolo migliore puo'
              pesare la meta' del portafoglio. E' il vero "non faccio
              niente", ed e' il confronto che rende evidente che cosa fa
              davvero il ribilanciamento.
    QUARTERLY, SEMIANNUALLY, ANNUALLY
              i pesi derivano dentro il periodo e vengono riportati a 1/n
              l'ultimo giorno di borsa del trimestre, del semestre o
              dell'anno.
    """
    tickers = [t for t in tickers if t in rendimenti.columns]
    if not tickers:
        return pd.Series(dtype=float)

    sotto = rendimenti[tickers].dropna()
    if sotto.empty:
        return pd.Series(dtype=float)

    n = len(tickers)
    R = sotto.to_numpy(dtype=np.float64)

    if metodo == "NONE":
        # Pesi costanti: il rendimento del portafoglio e' la media semplice
        # dei rendimenti dei titoli, giorno per giorno.
        return pd.Series(R.mean(axis=1), index=sotto.index, name="portafoglio")

    G = 1.0 + R
    T = R.shape[0]
    tagli = _confini_ribilanciamento(sotto.index, metodo)
    # Ogni segmento va da un ribilanciamento al successivo.
    inizi = np.concatenate(([0], tagli + 1))
    fini = np.concatenate((tagli + 1, [T]))

    valore = np.empty(T, dtype=np.float64)
    capitale = 1.0
    for a, b in zip(inizi, fini):
        if a >= b:
            continue
        # Dentro il segmento i pesi non si toccano: ogni titolo cresce per
        # conto proprio e il portafoglio vale la media delle quote.
        crescita = np.cumprod(G[a:b], axis=0)
        valore[a:b] = capitale * crescita.mean(axis=1)
        capitale = valore[b - 1]

    # Rendimenti giornalieri del capitale, primo giorno compreso.
    rendimento = np.empty(T, dtype=np.float64)
    rendimento[0] = valore[0] - 1.0
    rendimento[1:] = valore[1:] / valore[:-1] - 1.0
    return pd.Series(rendimento, index=sotto.index, name="portafoglio")


def equity(rendimenti: pd.Series, capitale_iniziale: float = 1.0) -> pd.Series:
    """Valore cumulato di un euro investito all'inizio."""
    if rendimenti.empty:
        return pd.Series(dtype=float)
    return capitale_iniziale * (1.0 + rendimenti).cumprod()


def drawdown(rendimenti: pd.Series) -> pd.Series:
    """Distanza dal massimo precedente, giorno per giorno."""
    curva = equity(rendimenti)
    if curva.empty:
        return pd.Series(dtype=float)
    return curva / curva.cummax() - 1.0


def misure(rendimenti: pd.Series, tasso_privo_rischio: float = 0.0) -> dict:
    """Le misure di sintesi di una serie di rendimenti giornalieri."""
    if rendimenti.empty:
        return {}
    curva = equity(rendimenti)
    anni = (curva.index[-1] - curva.index[0]).days / 365.25
    totale = float(curva.iloc[-1]) - 1.0
    vol = float(rendimenti.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    media = float(rendimenti.mean() * TRADING_DAYS_PER_YEAR)
    dd = drawdown(rendimenti)
    return {
        "giorni": int(len(rendimenti)),
        "inizio": curva.index[0],
        "fine": curva.index[-1],
        "anni": anni,
        "rendimento_totale": totale,
        "cagr": (1.0 + totale) ** (1.0 / anni) - 1.0 if anni > 0 else np.nan,
        "rendimento_medio_ann": media,
        "volatilita": vol,
        "sharpe": (media - tasso_privo_rischio) / vol if vol > 0 else np.nan,
        "max_drawdown": float(dd.min()) if not dd.empty else np.nan,
        "data_max_drawdown": dd.idxmin() if not dd.empty else None,
    }


# --------------------------------------------------------------------------
# Statistiche rolling
# --------------------------------------------------------------------------
def _rendimenti_rolling(cumulata: np.ndarray, finestra: int) -> np.ndarray:
    """Rendimento totale di ogni finestra di 'finestra' giorni consecutivi."""
    if cumulata.size <= finestra:
        return np.empty(0)
    return cumulata[finestra:] / cumulata[:-finestra] - 1.0


def _drawdown_rolling(valori: np.ndarray, finestra: int) -> np.ndarray:
    """
    Drawdown massimo dentro ogni finestra di 'finestra' giorni.

    E' il calcolo che nel notebook costava piu' di tutto il resto messo
    insieme: '.rolling(window).apply(qs.stats.max_drawdown)' chiamava una
    funzione Python una volta per ogni giorno di ogni finestra. Qui si usa
    una vista scorrevole di NumPy - che non copia nulla - e il massimo
    progressivo lungo la finestra, a blocchi per non occupare memoria.
    """
    n = valori.size
    if n < finestra:
        return np.empty(0)

    from numpy.lib.stride_tricks import sliding_window_view

    viste = sliding_window_view(valori, finestra)
    risultato = np.empty(viste.shape[0], dtype=np.float64)
    for inizio in range(0, viste.shape[0], BLOCCO_FINESTRE):
        blocco = viste[inizio:inizio + BLOCCO_FINESTRE]
        massimi = np.maximum.accumulate(blocco, axis=1)
        risultato[inizio:inizio + blocco.shape[0]] = (blocco / massimi).min(axis=1) - 1.0
    return risultato


def statistiche_rolling(rendimenti: pd.Series, finestre_anni: dict[str, int],
                        giorni_per_anno: int = TRADING_DAYS_PER_YEAR
                        ) -> pd.DataFrame:
    """
    Come si sarebbe comportato il portafoglio su ogni possibile periodo di N anni.

    E' la tabella piu' onesta del report. Il rendimento complessivo di
    vent'anni dipende in modo pesante da quando si e' entrati; questa tabella
    guarda TUTTI i possibili punti di ingresso e mostra il migliore, il
    peggiore e la mediana. Un portafoglio il cui peggior quinquennio e'
    positivo e' una cosa molto diversa da uno che ha lo stesso rendimento
    medio ma un quinquennio a meno trenta per cento.
    """
    colonne = ["Periodo", "Rend. medio", "Rend. minimo", "Rend. massimo",
               "Rend. mediano", "DD medio", "DD minimo", "DD massimo",
               "DD mediano", "Finestre"]
    if rendimenti.empty:
        return pd.DataFrame(columns=colonne)

    valori = (1.0 + rendimenti.to_numpy(dtype=np.float64)).cumprod()
    # Serie cumulata con un 1 davanti: permette di ottenere il rendimento di
    # qualunque intervallo come rapporto fra due soli numeri.
    cumulata = np.concatenate(([1.0], valori))

    righe = []
    for etichetta, anni in finestre_anni.items():
        finestra = int(anni * giorni_per_anno)
        if rendimenti.size < finestra or finestra <= 0:
            righe.append({"Periodo": etichetta, "Finestre": 0,
                          **{c: np.nan for c in colonne[1:-1]}})
            continue

        rend = _rendimenti_rolling(cumulata, finestra)
        dd = _drawdown_rolling(valori, finestra)

        righe.append({
            "Periodo": etichetta,
            "Rend. medio": float(np.mean(rend)),
            "Rend. minimo": float(np.min(rend)),
            "Rend. massimo": float(np.max(rend)),
            "Rend. mediano": float(np.median(rend)),
            # "DD minimo" e' il drawdown MENO severo fra tutte le finestre,
            # "DD massimo" il piu' severo: sono numeri negativi, e il piu'
            # severo e' il piu' piccolo. Le etichette seguono il notebook.
            "DD medio": float(np.mean(dd)),
            "DD minimo": float(np.max(dd)),
            "DD massimo": float(np.min(dd)),
            "DD mediano": float(np.median(dd)),
            "Finestre": int(rend.size),
        })

    return pd.DataFrame(righe, columns=colonne)


# --------------------------------------------------------------------------
# Correlazioni
# --------------------------------------------------------------------------
def titoli_piu_frequenti(sim: Simulazione, quanti_portafogli: int,
                         quanti_titoli: int) -> list[str]:
    """
    I titoli che compaiono piu' spesso nei portafogli con lo Sharpe migliore.

    Sono i "soliti noti" della selezione: guardare come sono correlati fra
    loro dice se la diversificazione del portafoglio e' vera o apparente.
    """
    df = sim.portafogli
    if df.empty:
        return []
    top = df.nlargest(min(quanti_portafogli, len(df)), "Sharpe_Ratio")
    conteggio: dict[str, int] = {}
    for riga in top.index:
        for t in sim.tickers(riga):
            conteggio[t] = conteggio.get(t, 0) + 1
    ordinati = sorted(conteggio.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ordinati[:quanti_titoli]]


def correlazioni(rendimenti: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Matrice di correlazione dei rendimenti giornalieri."""
    presenti = [t for t in tickers if t in rendimenti.columns]
    if len(presenti) < 2:
        return pd.DataFrame()
    return rendimenti[presenti].corr()
