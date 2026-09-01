"""
Prova che la versione veloce dia gli stessi risultati di quella del notebook.

E' il test che conta piu' di ogni altro. Riscrivere un calcolo per renderlo
cento volte piu' rapido serve a poco se poi seleziona titoli diversi: qui si
confrontano, su dati sintetici ma realistici, le decisioni dello screener
vettoriale con quelle delle funzioni originali della Cella 3, copiate
verbatim, e i numeri della simulazione con quelli del ciclo Python della
Cella 4.

Si esegue con:
    python -m pytest tests -q
oppure, senza pytest installato:
    python tests/test_equivalenza.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics, portfolio, screener              # noqa: E402
from src.config import TRADING_DAYS_PER_YEAR              # noqa: E402

GIORNI_ANNO = TRADING_DAYS_PER_YEAR


# ==========================================================================
# Le funzioni originali del notebook, copiate senza modifiche
# ==========================================================================
def ref_fast_max_drawdown(prices_array):
    running_max = np.maximum.accumulate(prices_array)
    running_max[running_max == 0] = 1
    drawdown = (prices_array / running_max) - 1
    return drawdown.min()


def ref_fast_sharpe(returns_array, rf, days):
    if len(returns_array) == 0:
        return -999.0
    mean_ret = np.mean(returns_array) * days
    std_dev = np.std(returns_array) * np.sqrt(days)
    if std_dev == 0:
        return -999.0
    return (mean_ret - rf) / std_dev


def ref_fast_volatility(prices_series):
    m_prices = prices_series.resample("ME").last()
    if len(m_prices) < 12:
        return -1.0
    p_arr = m_prices.values
    valid = p_arr[:-1] != 0
    if not np.any(valid):
        return -1.0
    m_ret = (p_arr[1:][valid] / p_arr[:-1][valid]) - 1
    return np.std(m_ret) * np.sqrt(12)


def ref_fast_check_rolling_neg_returns(prices_array, years_window):
    days_window = int(years_window * 252)
    n_days = len(prices_array)
    if n_days <= days_window:
        return False
    end_prices = prices_array[days_window:]
    start_prices = prices_array[:-days_window]
    valid_mask = start_prices > 0
    if not np.any(valid_mask):
        return False
    rolling_returns = (end_prices[valid_mask] / start_prices[valid_mask]) - 1
    return np.min(rolling_returns) >= 0


def ref_process_ticker(df, p):
    """La catena di filtri della Cella 3, nello stesso ordine."""
    if len(df) < GIORNI_ANNO:
        return "ERR_EMPTY"

    start_date, end_date = df.index.min(), df.index.max()
    if (end_date - start_date).days / 365.25 < (p["anni_storico"] - 0.1):
        return "FAIL_HISTORY_LEN"

    prices_array = df["adjusted_close"].values
    volume_array = df["volume"].values

    last_price = prices_array[-1]
    if last_price > p["prezzo_massimo"]:
        return "FAIL_PRICE_MAX"
    if last_price < p["prezzo_minimo"]:
        return "FAIL_PRICE_MIN"
    if last_price <= 0:
        return "FAIL_PRICE_ZERO"

    lookback_vol = 90
    if len(prices_array) > lookback_vol:
        recent_prices = prices_array[-lookback_vol:]
        recent_volumes = np.nan_to_num(volume_array[-lookback_vol:], nan=0.0)
        avg_dollar_volume = np.mean(recent_prices * recent_volumes)
        if avg_dollar_volume < p["dollar_volume_minimo"]:
            return "FAIL_LIQUIDITY"

    if not ref_fast_check_rolling_neg_returns(prices_array, p["anni_perf_continua"]):
        return "FAIL_PERF_ROLLING"

    valid_mask = prices_array[:-1] > 0
    if not np.any(valid_mask):
        return "ERR_DATA"
    daily_ret = (prices_array[1:][valid_mask] / prices_array[:-1][valid_mask]) - 1

    mdd = ref_fast_max_drawdown(prices_array)
    if mdd < -p["max_drawdown"]:
        return "FAIL_DD"

    sr = ref_fast_sharpe(daily_ret, p["tasso_privo_rischio"], GIORNI_ANNO)
    if sr < p["min_sharpe"]:
        return "FAIL_SR"

    vol = ref_fast_volatility(df["adjusted_close"])
    if vol < p["min_volatilita"]:
        return "FAIL_VOL"

    return "PASSED"


# I codici del notebook e quelli dell'applicazione dicono la stessa cosa
# con nomi diversi.
TRADUZIONE = {
    "PASSED": "PASSATO",
    "ERR_EMPTY": "DATI",
    "ERR_DATA": "DATI",
    "FAIL_HISTORY_LEN": "STORICO",
    "FAIL_PRICE_MAX": "PREZZO_MAX",
    "FAIL_PRICE_MIN": "PREZZO_MIN",
    "FAIL_PRICE_ZERO": "DATI",
    "FAIL_LIQUIDITY": "LIQUIDITA",
    "FAIL_PERF_ROLLING": "PERF_CONTINUA",
    "FAIL_DD": "DRAWDOWN",
    "FAIL_SR": "SHARPE",
    "FAIL_VOL": "VOLATILITA",
}


# ==========================================================================
# Dati sintetici
# ==========================================================================
def genera_titoli(quanti: int = 400, seme: int = 7) -> dict[str, pd.DataFrame]:
    """
    Titoli finti ma verosimili, con storie di lunghezza molto diversa.

    Si mescolano deriva e volatilita' in modo da avere di tutto: titoli che
    passano, titoli troppo giovani, titoli illiquidi, penny stock, titoli
    con drawdown devastanti e titoli piatti. Serve che ogni ramo della
    catena di filtri venga effettivamente percorso.
    """
    rng = np.random.default_rng(seme)
    fine = pd.Timestamp("2025-12-31")
    titoli = {}

    for i in range(quanti):
        anni = float(rng.choice([0.5, 2, 8, 15, 20.5, 22, 25],
                                p=[.08, .12, .15, .15, .2, .2, .1]))
        n = max(30, int(anni * GIORNI_ANNO))
        date = pd.bdate_range(end=fine, periods=n)

        deriva = rng.normal(0.00035, 0.00035)
        vol = float(rng.choice([0.004, 0.010, 0.020, 0.035]))
        shock = rng.normal(deriva, vol, size=n)
        prezzi = float(rng.choice([0.8, 3.0, 30.0, 400.0, 6000.0])) \
            * np.exp(np.cumsum(shock))

        volumi = np.abs(rng.normal(
            float(rng.choice([1e3, 5e4, 2e6, 3e7])), 1e4, size=n))
        if rng.random() < 0.05:
            volumi[:] = 0.0                     # titolo che non scambia

        titoli[f"T{i:04d}.US"] = pd.DataFrame(
            {"adjusted_close": prezzi.astype(np.float32),
             "volume": volumi.astype(np.float32)},
            index=pd.DatetimeIndex(date, name="date"),
        )
    return titoli


PARAMETRI = [
    dict(anni_storico=20, anni_perf_continua=5, max_drawdown=0.65,
         min_sharpe=0.5, min_volatilita=0.0, prezzo_minimo=5.0,
         prezzo_massimo=5000.0, dollar_volume_minimo=10_000_000.0,
         tasso_privo_rischio=0.0),
    dict(anni_storico=15, anni_perf_continua=3, max_drawdown=0.40,
         min_sharpe=0.0, min_volatilita=0.10, prezzo_minimo=1.0,
         prezzo_massimo=1000.0, dollar_volume_minimo=1_000_000.0,
         tasso_privo_rischio=0.02),
    dict(anni_storico=10, anni_perf_continua=1, max_drawdown=0.90,
         min_sharpe=-1.0, min_volatilita=0.0, prezzo_minimo=0.0,
         prezzo_massimo=100_000.0, dollar_volume_minimo=0.0,
         tasso_privo_rischio=0.0),
]


# ==========================================================================
# I test
# ==========================================================================
def test_screener_da_gli_stessi_esiti():
    """Ogni titolo riceve lo stesso verdetto della Cella 3, su tre set di soglie."""
    titoli = genera_titoli()

    righe = []
    for ticker, df in titoli.items():
        riga = metrics.misura(
            ticker,
            df.index.to_numpy(dtype="datetime64[D]"),
            df["adjusted_close"].to_numpy(dtype=np.float32),
            df["volume"].to_numpy(dtype=np.float32),
        )
        righe.append(riga if riga is not None else metrics.riga_vuota(ticker))
    metriche = pd.DataFrame(righe)

    for parametri in PARAMETRI:
        attesi = {t: TRADUZIONE[ref_process_ticker(df, parametri)]
                  for t, df in titoli.items()}
        ottenuti = dict(zip(
            *screener.applica(metriche, screener.Filtri(**parametri))
            [["ticker", "esito"]].to_numpy().T
        ))

        diversi = {t: (attesi[t], ottenuti[t])
                   for t in attesi if attesi[t] != ottenuti[t]}
        assert not diversi, (
            f"Verdetti diversi con {parametri}:\n"
            + "\n".join(f"  {t}: notebook={a}, app={b}"
                        for t, (a, b) in list(diversi.items())[:15])
        )

        promossi = sum(1 for v in attesi.values() if v == "PASSATO")
        print(f"  soglie anni={parametri['anni_storico']:>2} "
              f"sharpe>={parametri['min_sharpe']:>4}: "
              f"{promossi} promossi su {len(attesi)}, verdetti identici")


def test_monte_carlo_identico_al_ciclo():
    """Rendimento, volatilita' e Sharpe coincidono con il calcolo uno-per-uno."""
    rng = np.random.default_rng(11)
    n = 60
    mu = rng.normal(0.10, 0.05, n)
    A = rng.normal(0, 1, (n, n))
    cov = (A @ A.T) / n * 0.04

    sim = portfolio.simula(mu, cov, [f"X{i}" for i in range(n)],
                           k_min=5, k_max=12, n_simulazioni=4000,
                           tasso_privo_rischio=0.01, seme=3)

    df = sim.portafogli
    for riga in rng.choice(len(df), size=300, replace=False):
        idx = sim.indici[riga]
        idx = idx[idx >= 0]
        k = idx.size
        assert k == df.at[riga, "Num_Tickers"]
        assert np.unique(idx).size == k, "titoli ripetuti nello stesso portafoglio"

        pesi = np.ones(k) / k
        atteso_ret = float(pesi @ mu[idx])
        atteso_vol = float(np.sqrt(pesi @ cov[np.ix_(idx, idx)] @ pesi))
        atteso_sr = (atteso_ret - 0.01) / atteso_vol

        assert abs(atteso_ret - df.at[riga, "Return"]) < 1e-10
        assert abs(atteso_vol - df.at[riga, "Volatility"]) < 1e-10
        assert abs(atteso_sr - df.at[riga, "Sharpe_Ratio"]) < 1e-10

    print(f"  {len(df)} portafogli simulati in {sim.secondi:.2f}s, "
          "300 verificati uno a uno")


def test_statistiche_rolling_identiche():
    """Le finestre mobili coincidono con .rolling().apply() del notebook."""
    rng = np.random.default_rng(5)
    n = 3000
    serie = pd.Series(rng.normal(0.0004, 0.011, n),
                      index=pd.bdate_range("2010-01-04", periods=n))

    def ref_max_drawdown(x):
        prezzi = (1 + x).cumprod()
        return (prezzi / prezzi.expanding(min_periods=0).max()).min() - 1

    finestre = {"1 anno": 1, "3 anni": 3}
    ottenute = portfolio.statistiche_rolling(serie, finestre)

    for etichetta, anni in finestre.items():
        w = anni * GIORNI_ANNO
        rif_rend = serie.rolling(w).apply(lambda x: (1 + x).prod() - 1,
                                          raw=False).dropna()
        rif_dd = serie.rolling(w).apply(ref_max_drawdown, raw=False).dropna()

        riga = ottenute.loc[ottenute["Periodo"] == etichetta].iloc[0]
        assert int(riga["Finestre"]) == len(rif_rend)
        for etichetta_col, atteso in [
            ("Rend. medio", rif_rend.mean()),
            ("Rend. minimo", rif_rend.min()),
            ("Rend. massimo", rif_rend.max()),
            ("Rend. mediano", rif_rend.median()),
            ("DD medio", rif_dd.mean()),
            ("DD minimo", rif_dd.max()),
            ("DD massimo", rif_dd.min()),
            ("DD mediano", rif_dd.median()),
        ]:
            assert abs(float(riga[etichetta_col]) - float(atteso)) < 1e-9, (
                f"{etichetta} / {etichetta_col}: "
                f"{riga[etichetta_col]} contro {atteso}"
            )
        print(f"  finestra {etichetta}: {len(rif_rend)} osservazioni, "
              "otto statistiche identiche")


def test_ribilanciamento_coerente():
    """
    Pesi costanti, buy & hold e ribilanciamenti periodici sono coerenti fra loro.

    Non c'e' un riferimento del notebook da replicare - il ciclo originale
    saltava in silenzio i ribilanciamenti caduti di sabato - ma le
    proprieta' devono valere: su un solo titolo tutti i metodi coincidono,
    e su piu' titoli il buy & hold deve derivare dai pesi iniziali.
    """
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    rend = pd.DataFrame(rng.normal(0.0004, 0.012, (len(idx), 4)),
                        index=idx, columns=list("ABCD"))

    uno = [portfolio.serie_portafoglio(rend, ["A"], m)
           for m in ("NONE", "BUYHOLD", "QUARTERLY", "ANNUALLY")]
    for s in uno[1:]:
        assert np.allclose(s.to_numpy(), uno[0].to_numpy(), atol=1e-12)

    tutti = ["A", "B", "C", "D"]
    bh = portfolio.serie_portafoglio(rend, tutti, "BUYHOLD")
    atteso = (1 + rend[tutti]).cumprod().mean(axis=1)
    assert abs(float(portfolio.equity(bh).iloc[-1]) - float(atteso.iloc[-1])) < 1e-9

    for metodo in ("NONE", "BUYHOLD", "QUARTERLY", "SEMIANNUALLY", "ANNUALLY"):
        s = portfolio.serie_portafoglio(rend, tutti, metodo)
        assert len(s) == len(rend), f"{metodo}: lunghezza {len(s)}"
        assert np.isfinite(s.to_numpy()).all()
    print("  cinque metodi di ribilanciamento coerenti su 1500 giorni")


if __name__ == "__main__":
    for nome, funzione in list(globals().items()):
        if nome.startswith("test_") and callable(funzione):
            print(f"\n{nome}")
            funzione()
    print("\nTutti i controlli superati.")
