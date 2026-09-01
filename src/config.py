"""
Costanti, parametri di default e palette dello Screener Markowitz.

Tutto cio' che un docente puo' voler cambiare senza toccare la logica sta
qui: soglie iniziali dei filtri, borse disponibili, percorsi dei file di
dato, colori dei grafici.

Il modulo non importa streamlit: viene usato anche dallo script di
costruzione dell'archivio da riga di comando (scripts/build_dataset.py),
che gira su GitHub Actions senza alcuna interfaccia.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Identita' dell'applicazione
# --------------------------------------------------------------------------
APP_TITLE = "Screener & Frontiera Efficiente"
APP_SUBTITLE = "Scuola di Finanza Operativa - Percorso Investing"
APP_COURSE = "Corso 2: Portafogli Avanzati"
APP_VERSION = "1.0"

# --------------------------------------------------------------------------
# Percorsi dell'archivio
# --------------------------------------------------------------------------
# La cartella dei dati puo' essere spostata con la variabile d'ambiente
# KQ_DATA_DIR: utile su Colab (per puntare a Google Drive) o in aula.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("KQ_DATA_DIR", PROJECT_ROOT / "data"))

# L'archivio e' fatto di quattro pezzi, con ruoli ben distinti:
#
#   meta.json        quando e come e' stato costruito
#   universe.parquet la lista completa dei simboli (codice, nome, tipo, borsa)
#   metrics.parquet  UNA RIGA PER OGNI TICKER DELL'UNIVERSO, con tutte le
#                    misure che servono ai filtri gia' calcolate. E' il file
#                    su cui lavora lo screener: ventitremila righe, sei
#                    megabyte, filtrabile in millisecondi.
#   close/part_NN.parquet
#                    le serie storiche dei prezzi, TUTTE, divise in fette da
#                    circa millecinquecento titoli l'una. La divisione non e'
#                    un vezzo: un unico parquet da ventitremila colonne ha un
#                    indice interno enorme e lento da aprire, mentre cosi'
#                    l'app scarica e legge solo le fette che contengono i
#                    titoli sopravvissuti ai filtri - qualche decina di
#                    megabyte invece di duecento.
FILE_META = "meta.json"
FILE_UNIVERSE = "universe.parquet"
FILE_METRICS = "metrics.parquet"
CLOSE_DIR = "close"
SHARD_PATTERN = "part_{:02d}.parquet"

# Quanti titoli per fetta del pannello prezzi.
SHARD_SIZE = 1500

# I file leggeri, che l'app scarica sempre e subito: bastano a far
# funzionare lo screener sull'universo intero.
ARCHIVE_CORE_FILES = (FILE_META, FILE_UNIVERSE, FILE_METRICS)

# Indirizzo da cui l'app scarica l'archivio se non lo trova in locale.
# Si imposta nei secrets di Streamlit come ARCHIVE_URL, oppure qui sotto.
# Deve essere il prefisso degli asset di una release di GitHub, per esempio:
#   https://github.com/UTENTE/REPO/releases/download/archivio-2026-09
# L'app vi appende il nome del file: .../meta.json, .../close/part_03.parquet
DEFAULT_ARCHIVE_URL = ""

# --------------------------------------------------------------------------
# Cadenza di aggiornamento
# --------------------------------------------------------------------------
# L'archivio si ricostruisce da zero due volte l'anno. Non si aggiorna in
# modo incrementale, e la ragione e' sostanziale: l'adjusted_close viene
# ricalcolato all'indietro su TUTTA la serie ogni volta che c'e' un
# frazionamento o uno stacco di dividendo. Accodare gli ultimi sei mesi a una
# serie vecchia produrrebbe un salto artificiale nel punto di giunzione.
ARCHIVE_MAX_AGE_DAYS = 180

# --------------------------------------------------------------------------
# Borse ed universo
# --------------------------------------------------------------------------
# Chiave: codice usato nell'URL /exchange-symbol-list/{CODICE}
# Valore: suffisso da appendere al ticker per le chiamate EOD.
# La mappatura non e' banale: la lista di Borsa Italiana si chiama "MIL" ma
# il suffisso dei dati e' ".MI", Xetra vuole ".XETRA" e non ".DE", Londra
# ".LSE" e non ".L". Questi accoppiamenti sono verificati sul campo.
EXCHANGE_CONFIG_USA = {
    "US": ".US",        # NYSE, NASDAQ, AMEX, ARCA in un'unica lista
}

EXCHANGE_CONFIG_GLOBAL = {
    "US": ".US",
    "XETRA": ".XETRA",   # Germania
    "PA": ".PA",         # Euronext Parigi
    "MIL": ".MI",        # Borsa Italiana
    "LSE": ".LSE",       # Londra
    "MC": ".MC",         # Madrid
    "TO": ".TO",         # Toronto
    "HK": ".HK",         # Hong Kong
}

UNIVERSE_CHOICES = {
    "USA": ("Solo Stati Uniti (NYSE, NASDAQ, AMEX, ETF)", EXCHANGE_CONFIG_USA),
    "GLOBAL": ("Globale (USA + Europa + Asia)", EXCHANGE_CONFIG_GLOBAL),
}

# Tipi di strumento ammessi nella lista simboli EODHD.
# Si escludono azioni privilegiate, warrant, fondi chiusi e obbligazioni.
ALLOWED_TYPES = ("Common Stock", "ETF")

BENCHMARK = "SPY.US"

# --------------------------------------------------------------------------
# Costruzione dell'archivio
# --------------------------------------------------------------------------
# NESSUN pre-filtro: si scarica e si misura l'universo INTERO - circa
# ventitremilacinquecento simboli sul solo mercato americano - esattamente
# come faceva il notebook. Il costo di questa scelta, una chiamata API per
# titolo, viene pagato una volta ogni sei mesi da una macchina di GitHub e
# non dallo studente davanti allo schermo.
#
# A novecento chiamate al minuto l'universo americano si scarica in circa
# mezz'ora. Il limite di EODHD e' di norma mille chiamate al minuto: si sta
# volutamente sotto, perche' un 429 costa piu' di quanto si guadagni a
# correre. Chi ha un piano piu' generoso puo' alzare il ritmo con la
# variabile d'ambiente KQ_RATE_PER_MIN.
DOWNLOAD_WORKERS = 16
DOWNLOAD_RATE_PER_MIN = int(os.environ.get("KQ_RATE_PER_MIN", 900))
DOWNLOAD_TIMEOUT = 25
DOWNLOAD_RETRIES = 2

# Sotto un anno di quotazioni un titolo non e' misurabile: e' lo stesso
# controllo che il notebook faceva con "len(df) < 252".
MIN_OBSERVATIONS = 252

# Finestra usata dal filtro di liquidita' (giorni di borsa).
LIQUIDITY_LOOKBACK = 90

# Finestre aggiuntive del volume in dollari, calcolate in fase di
# costruzione e mostrate a video come contorno informativo.
DOLLARVOL_WINDOWS = (30, 90, 200)

# --------------------------------------------------------------------------
# Parametri di default dello screener (gli stessi del notebook Colab)
# --------------------------------------------------------------------------
DEFAULT_YEARS_HISTORY = 20            # anni di storico richiesti
DEFAULT_YEARS_NO_NEG = 5              # finestra della performance continua
DEFAULT_MAX_DRAWDOWN = 0.65           # drawdown massimo tollerato sul singolo titolo
DEFAULT_MIN_SHARPE = 0.5
DEFAULT_MIN_VOL = 0.00                # volatilita' annualizzata minima
DEFAULT_MAX_PRICE = 5000.0
DEFAULT_MIN_PRICE = 5.0               # sotto i 5 dollari si entra nelle penny stock
DEFAULT_MIN_DOLLAR_VOLUME = 10_000_000.0

# Finestre della performance continua selezionabili a video.
ROLLING_YEARS_CHOICES = tuple(range(1, 11))
MAX_ROLLING_YEARS = 10

# --------------------------------------------------------------------------
# Parametri di default della simulazione
# --------------------------------------------------------------------------
DEFAULT_MIN_TICKERS = 5
DEFAULT_MAX_TICKERS = 10
DEFAULT_NUM_SIMULATIONS = 50_000
DEFAULT_RISK_FREE = 0.0
DEFAULT_REBALANCE = "NONE"
DEFAULT_SEED = 42

# Vincolo di diversificazione del portafoglio "massimo rendimento":
# senza un numero minimo di titoli vincerebbe sempre il piu' concentrato.
MIN_TICKERS_FOR_MAX_RETURN = 3

REBALANCE_METHODS = {
    "NONE": "Pesi costanti (ribilanciamento giornaliero)",
    "BUYHOLD": "Buy & hold (nessun ribilanciamento)",
    "QUARTERLY": "Trimestrale",
    "SEMIANNUALLY": "Semestrale",
    "ANNUALLY": "Annuale",
}
REBALANCE_ORDER = ["NONE", "BUYHOLD", "QUARTERLY", "SEMIANNUALLY", "ANNUALLY"]

TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12

# Tolleranza sull'anzianita' dello storico in fase di simulazione: si
# accettano titoli che partono fino a sessanta giorni dopo la data limite,
# perche' i calendari di borsa non coincidono mai al giorno esatto.
HISTORY_TOLERANCE_DAYS = 60

# Finestre delle statistiche rolling mostrate nel report (in anni).
ROLLING_WINDOWS = {
    "1 anno": 1,
    "2 anni": 2,
    "3 anni": 3,
    "5 anni": 5,
}

# Quanti punti disegnare al massimo nel grafico della frontiera efficiente.
# Cinquantamila punti in un grafico interattivo bloccano il browser; il
# campione mostrato resta rappresentativo e le statistiche usano tutti i dati.
FRONTIER_MAX_POINTS = 12_000

# Quanti titoli mostrare nella mappa delle correlazioni.
HEATMAP_TOP_PORTFOLIOS = 50
HEATMAP_MAX_TICKERS = 15

# --------------------------------------------------------------------------
# Palette grafica (identica a quella del Simulatore di Portafogli)
# --------------------------------------------------------------------------
COLORS = {
    "bg": "#0B1220",
    "panel": "#141F35",
    "grid": "#22314F",
    "text": "#E6EDF7",
    "muted": "#8FA3C0",
    "accent": "#2DD4BF",      # teal - identita' dell'app
    "accent2": "#60A5FA",     # azzurro
    "positive": "#34D399",
    "negative": "#F87171",
    "warning": "#FBBF24",
    "satellite": "#F97316",
}

NAME_MAX_SHARPE = "Max Sharpe"
NAME_MIN_VOL = "Minima volatilita"
NAME_MAX_RET = "Massimo rendimento"
NAME_BENCHMARK = "SPY (benchmark)"

# Un colore per ciascun portafoglio chiave, usato in tutti i grafici.
STRATEGY_COLORS = {
    NAME_MAX_SHARPE: "#2DD4BF",
    NAME_MIN_VOL: "#60A5FA",
    NAME_MAX_RET: "#A78BFA",
    NAME_BENCHMARK: "#8FA3C0",
}
STRATEGY_ORDER = [NAME_MAX_SHARPE, NAME_MIN_VOL, NAME_MAX_RET]

PLOTLY_TEMPLATE = "plotly_dark"

# --------------------------------------------------------------------------
# Motivi di scarto dello screener
# --------------------------------------------------------------------------
REJECT_LABELS = {
    "PASSATO": "Superano tutti i filtri",
    "DATI": "Dati insufficienti o non validi",
    "STORICO": "Storico troppo corto",
    "PREZZO_MAX": "Prezzo sopra la soglia massima",
    "PREZZO_MIN": "Prezzo sotto la soglia minima",
    "LIQUIDITA": "Volume in dollari insufficiente",
    "PERF_CONTINUA": "Almeno un periodo pluriennale in perdita",
    "DRAWDOWN": "Drawdown storico troppo profondo",
    "SHARPE": "Sharpe ratio sotto la soglia",
    "VOLATILITA": "Volatilita' sotto la soglia",
}

# L'ordine conta: e' quello in cui il notebook applicava i filtri, e quindi
# quello in cui un titolo viene attribuito al primo motivo che lo scarta.
REJECT_ORDER = [
    "DATI",
    "STORICO",
    "PREZZO_MAX",
    "PREZZO_MIN",
    "LIQUIDITA",
    "PERF_CONTINUA",
    "DRAWDOWN",
    "SHARPE",
    "VOLATILITA",
]
