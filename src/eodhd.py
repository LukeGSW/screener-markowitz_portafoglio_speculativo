"""
Client per le API di EOD Historical Data (EODHD).

Questo modulo serve due padroni con esigenze opposte:

  * lo script di costruzione dell'archivio, che gira su GitHub Actions senza
    interfaccia e deve scaricare ventitremila serie storiche il piu' in
    fretta possibile senza farsi bloccare;
  * l'applicazione Streamlit, che in condizioni normali NON chiama affatto
    EODHD, perche' legge l'archivio gia' pronto.

Per questo non importa streamlit a livello di modulo: l'import e' tentato e,
se fallisce, il modulo continua a funzionare.

Il guadagno di velocita' rispetto al notebook sta in tre dettagli:

  1. una sola sessione HTTP per thread, riusata per tutte le chiamate. Il
     notebook apriva "with requests.Session() as s" dentro ogni download:
     una stretta di mano TLS nuova per ciascuno dei ventitremila titoli,
     che da sola vale piu' del tempo di risposta dell'API;
  2. un limitatore a gettoni condiviso, che tiene il ritmo appena sotto il
     tetto del piano invece di alternare raffiche e blocchi 429;
  3. i risultati vengono consegnati a mano a mano che arrivano, cosi' chi
     chiama puo' misurarli e buttare via i dati grezzi senza accumulare in
     memoria ventitremila DataFrame.

Endpoint utilizzati
-------------------
  GET /api/exchange-symbol-list/{BORSA}   elenco dei simboli quotati
  GET /api/eod/{SIMBOLO}                  serie storica End-Of-Day
  GET /api/user                           stato dell'abbonamento
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from .config import (ALLOWED_TYPES, DOWNLOAD_RATE_PER_MIN, DOWNLOAD_RETRIES,
                     DOWNLOAD_TIMEOUT, DOWNLOAD_WORKERS)

BASE_URL = "https://eodhd.com/api"


class EodhdError(RuntimeError):
    """Errore riconducibile alle API EODHD, con messaggio gia' in italiano."""


# --------------------------------------------------------------------------
# Chiave API
# --------------------------------------------------------------------------
_PLACEHOLDERS = {"", "INCOLLA_QUI_LA_CHIAVE", "LA_TUA_API_KEY_EODHD", "demo"}


def get_api_key() -> str | None:
    """
    Legge la chiave dai secrets di Streamlit, o dall'ambiente.

    Su GitHub Actions arriva dalla variabile d'ambiente EODHD_API_KEY, che il
    workflow riempie dal segreto del repository. Nell'app pubblicata arriva
    dai secrets di Streamlit. In nessuno dei due casi compare a video.
    """
    key = None
    try:  # streamlit puo' non essere installato (script da riga di comando)
        import streamlit as st

        key = st.secrets.get("EODHD_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("EODHD_API_KEY")
    if isinstance(key, str):
        key = key.strip()
    if not key or key in _PLACEHOLDERS:
        return None
    return key


def mask_key(key: str | None) -> str:
    """Rappresentazione oscurata, per le schermate di diagnostica."""
    if not key:
        return "non impostata"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


# --------------------------------------------------------------------------
# Limitatore a gettoni
# --------------------------------------------------------------------------
class RateLimiter:
    """
    Distribuisce nel tempo le chiamate di piu' thread.

    Non e' una finestra scorrevole: e' un secchio che si riempie a ritmo
    costante e da cui ogni chiamata preleva un gettone. Il risultato e' un
    flusso regolare - niente raffiche iniziali seguite da un muro di errori
    429 - che a parita' di tetto e' anche piu' veloce, perche' nessuna
    chiamata viene sprecata e poi ripetuta.
    """

    def __init__(self, per_minute: int):
        self.rate = max(float(per_minute), 1.0) / 60.0   # gettoni al secondo
        self.capacity = max(float(per_minute) / 10.0, 1.0)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                attesa = (1.0 - self._tokens) / self.rate
            time.sleep(min(attesa, 0.5))


# --------------------------------------------------------------------------
# Sessioni HTTP
# --------------------------------------------------------------------------
_thread_local = threading.local()


def _session() -> requests.Session:
    """Una sessione per thread, con un pool di connessioni gia' aperte."""
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=DOWNLOAD_WORKERS,
            pool_maxsize=DOWNLOAD_WORKERS,
            max_retries=0,          # i tentativi li gestiamo noi, con criterio
        )
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        sess.headers.update({"User-Agent": "KriterionQuant-Screener/1.0"})
        _thread_local.session = sess
    return sess


def _explain(status: int, symbol: str = "") -> str:
    dove = f" ({symbol})" if symbol else ""
    if status == 401:
        return ("Chiave API EODHD non valida o mancante (401)"
                f"{dove}. Verifica il segreto EODHD_API_KEY.")
    if status == 402:
        return f"Monte chiamate giornaliero di EODHD esaurito (402){dove}."
    if status == 403:
        return ("Accesso negato da EODHD (403)"
                f"{dove}: il simbolo o il mercato non e' incluso nel piano.")
    if status == 404:
        return f"Simbolo non trovato su EODHD (404){dove}."
    if status == 429:
        return f"Superato il limite di chiamate al minuto (429){dove}."
    return f"EODHD ha risposto con il codice {status}{dove}."


def _get(path: str, token: str, params: dict, timeout: int,
         limiter: RateLimiter | None = None) -> requests.Response:
    if limiter is not None:
        limiter.acquire()
    params = dict(params)
    params["api_token"] = token
    params.setdefault("fmt", "json")
    return _session().get(f"{BASE_URL}/{path}", params=params, timeout=timeout)


# --------------------------------------------------------------------------
# Stato dell'abbonamento
# --------------------------------------------------------------------------
def fetch_user_info(token: str) -> dict:
    """Chiamate consumate, tetto giornaliero, scadenza del piano."""
    resp = _get("user", token, {}, DOWNLOAD_TIMEOUT)
    if resp.status_code != 200:
        raise EodhdError(_explain(resp.status_code))
    try:
        data = resp.json()
    except ValueError:
        raise EodhdError("Risposta non interpretabile dall'endpoint /user.")
    if not isinstance(data, dict):
        raise EodhdError("Risposta inattesa dall'endpoint /user di EODHD.")
    return data


# --------------------------------------------------------------------------
# Universo dei simboli
# --------------------------------------------------------------------------
def fetch_exchange_symbols(exchange_code: str, token: str) -> list[dict]:
    """
    Elenco completo dei simboli quotati su una borsa.

    Una sola chiamata restituisce l'intero listino: e' cosi' che si arriva a
    ventitremila titoli americani senza ventitremila richieste.
    """
    resp = _get(f"exchange-symbol-list/{exchange_code}", token,
                {}, max(DOWNLOAD_TIMEOUT, 60))
    if resp.status_code != 200:
        raise EodhdError(_explain(resp.status_code, exchange_code))
    try:
        data = resp.json()
    except ValueError:
        raise EodhdError(f"Elenco simboli di '{exchange_code}' illeggibile.")
    if not isinstance(data, list):
        raise EodhdError(f"Elenco simboli di '{exchange_code}' in formato inatteso.")
    return data


def build_universe(exchange_config: dict[str, str], token: str,
                   benchmark: str | None = None,
                   on_progress: Callable[[str, int], None] | None = None,
                   ) -> list[dict]:
    """
    Costruisce l'universo applicando a ogni borsa il suffisso corretto.

    Tiene solo azioni ordinarie ed ETF, come il notebook: fuori restano
    azioni privilegiate, warrant, fondi chiusi e obbligazioni, che non hanno
    senso in un esercizio di frontiera efficiente su titoli azionari.

    Restituisce una lista di dizionari con 'ticker', 'code', 'name', 'type',
    'exchange', ordinata per ticker e senza duplicati.
    """
    visti: dict[str, dict] = {}

    for ex_code, suffix in exchange_config.items():
        righe = fetch_exchange_symbols(ex_code, token)
        contati = 0
        for item in righe:
            code = str(item.get("Code", "") or "").strip().upper()
            if not code:
                continue
            if item.get("Type") not in ALLOWED_TYPES:
                continue
            ticker = code if code.endswith(suffix) else f"{code}{suffix}"
            if ticker in visti:
                continue
            visti[ticker] = {
                "ticker": ticker,
                "code": code,
                "name": str(item.get("Name", "") or "")[:120],
                "type": str(item.get("Type", "") or ""),
                "exchange": ex_code,
                "currency": str(item.get("Currency", "") or ""),
            }
            contati += 1
        if on_progress is not None:
            on_progress(ex_code, contati)

    if benchmark and benchmark not in visti:
        visti[benchmark] = {
            "ticker": benchmark, "code": benchmark.split(".")[0],
            "name": "Benchmark", "type": "ETF",
            "exchange": benchmark.split(".")[-1], "currency": "USD",
        }

    return [visti[t] for t in sorted(visti)]


# --------------------------------------------------------------------------
# Serie storiche
# --------------------------------------------------------------------------
@dataclass(slots=True)
class SerieGrezza:
    """Una serie EOD appena scaricata, in array NumPy e senza pandas."""

    ticker: str
    stato: str                  # 'OK', 'VUOTO', 'ASSENTE', 'ERRORE: ...'
    date: np.ndarray | None = None    # datetime64[D]
    close: np.ndarray | None = None   # float32, adjusted_close
    volume: np.ndarray | None = None  # float32


def fetch_eod(ticker: str, token: str, start: str, end: str,
              limiter: RateLimiter | None = None,
              tentativi: int = DOWNLOAD_RETRIES) -> SerieGrezza:
    """
    Scarica la serie End-Of-Day di un titolo e la restituisce in array NumPy.

    Non solleva eccezioni per i casi ordinari (titolo inesistente, serie
    vuota): li segnala nel campo 'stato', perche' su ventitremila titoli
    qualche migliaio di buchi e' la norma e non deve fermare il lavoro.
    """
    params = {"period": "d", "from": start, "to": end, "order": "a"}

    ultimo_errore = ""
    for tentativo in range(tentativi + 1):
        try:
            resp = _get(f"eod/{ticker}", token, params, DOWNLOAD_TIMEOUT, limiter)
        except requests.exceptions.RequestException as exc:
            ultimo_errore = f"rete: {type(exc).__name__}"
            time.sleep(0.5 * (tentativo + 1))
            continue

        if resp.status_code == 404:
            return SerieGrezza(ticker, "ASSENTE")
        if resp.status_code in (429, 500, 502, 503, 504):
            # Rallenta e riprova: sono errori transitori.
            ultimo_errore = _explain(resp.status_code, ticker)
            time.sleep(1.0 + tentativo)
            continue
        if resp.status_code in (401, 402, 403):
            # Errori che non passeranno mai da soli: inutile insistere.
            raise EodhdError(_explain(resp.status_code, ticker))
        if resp.status_code != 200:
            ultimo_errore = _explain(resp.status_code, ticker)
            time.sleep(0.5)
            continue

        try:
            righe = resp.json()
        except ValueError:
            ultimo_errore = "risposta non interpretabile"
            continue

        if not isinstance(righe, list) or not righe:
            return SerieGrezza(ticker, "VUOTO")

        return _to_arrays(ticker, righe)

    return SerieGrezza(ticker, f"ERRORE: {ultimo_errore}")


def _to_arrays(ticker: str, righe: list[dict]) -> SerieGrezza:
    """Converte la risposta JSON in tre array allineati, gia' ripuliti."""
    n = len(righe)
    date = np.empty(n, dtype="datetime64[D]")
    close = np.empty(n, dtype=np.float64)
    volume = np.empty(n, dtype=np.float64)

    valide = 0
    for r in righe:
        d = r.get("date")
        if not d:
            continue
        # L'adjusted_close e' la serie corretta per dividendi e frazionamenti:
        # e' l'unica su cui abbia senso misurare un rendimento pluriennale.
        c = r.get("adjusted_close")
        if c is None:
            c = r.get("close")
        if c is None:
            continue
        try:
            date[valide] = np.datetime64(str(d)[:10])
            close[valide] = float(c)
        except (ValueError, TypeError):
            continue
        v = r.get("volume")
        try:
            volume[valide] = float(v) if v is not None else 0.0
        except (ValueError, TypeError):
            volume[valide] = 0.0
        valide += 1

    if valide == 0:
        return SerieGrezza(ticker, "VUOTO")

    date = date[:valide]
    close = close[:valide]
    volume = volume[:valide]

    # Ordine crescente e niente date ripetute (capita sui titoli piu' vecchi).
    ordine = np.argsort(date, kind="stable")
    date, close, volume = date[ordine], close[ordine], volume[ordine]
    if date.size > 1:
        tieni = np.ones(date.size, dtype=bool)
        tieni[:-1] = date[:-1] != date[1:]     # a parita' di data vince l'ultima
        date, close, volume = date[tieni], close[tieni], volume[tieni]

    # I prezzi non positivi o non finiti non sono prezzi.
    buoni = np.isfinite(close) & (close > 0)
    if not buoni.any():
        return SerieGrezza(ticker, "VUOTO")
    date, close, volume = date[buoni], close[buoni], volume[buoni]

    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

    return SerieGrezza(ticker, "OK",
                       date=date,
                       close=close.astype(np.float32),
                       volume=volume.astype(np.float32))


# --------------------------------------------------------------------------
# Download parallelo in streaming
# --------------------------------------------------------------------------
def download_series(tickers: Sequence[str], token: str, start: str, end: str,
                    workers: int = DOWNLOAD_WORKERS,
                    per_minute: int = DOWNLOAD_RATE_PER_MIN,
                    ) -> Iterator[SerieGrezza]:
    """
    Scarica in parallelo e consegna i risultati a mano a mano che arrivano.

    E' un generatore, non una lista: chi chiama misura la serie, tiene quel
    poco che gli serve e lascia che il resto venga raccolto dal garbage
    collector. E' l'unico modo per non arrivare a fine download con
    ventitremila serie storiche tutte in memoria.
    """
    limiter = RateLimiter(per_minute)
    tickers = list(tickers)
    if not tickers:
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futuri = {
            pool.submit(fetch_eod, t, token, start, end, limiter): t
            for t in tickers
        }
        try:
            for futuro in concurrent.futures.as_completed(futuri):
                ticker = futuri[futuro]
                try:
                    yield futuro.result()
                except EodhdError:
                    # Errore definitivo (chiave, quota, permessi): si ferma
                    # tutto, perche' insistere consumerebbe soltanto tempo.
                    for f in futuri:
                        f.cancel()
                    raise
                except Exception as exc:
                    yield SerieGrezza(ticker, f"ERRORE: {exc}")
        finally:
            for f in futuri:
                f.cancel()


def stima_durata(n_ticker: int, per_minute: int = DOWNLOAD_RATE_PER_MIN) -> float:
    """Minuti attesi per scaricare n titoli al ritmo dato."""
    if per_minute <= 0:
        return float("inf")
    return n_ticker / float(per_minute)
