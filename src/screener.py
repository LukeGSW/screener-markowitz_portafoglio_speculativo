"""
Lo screener: dalle misure alle soglie.

Tutto il lavoro pesante e' gia' stato fatto in fase di costruzione
dell'archivio (vedi metrics.py). Qui resta soltanto da confrontare colonne
di numeri con le soglie scelte dallo studente, il che su ventitremila righe
richiede qualche millisecondo. E' per questo che nell'applicazione i cursori
rispondono all'istante mentre nel notebook ogni cambio di parametro
significava rileggere l'intero disco.

L'ordine dei filtri e' quello del notebook, e conta: a ogni titolo viene
attribuito il PRIMO motivo che lo esclude, cosi' il riepilogo dice davvero
"quanti titoli sono caduti sulla liquidita'" e non "quanti titoli sarebbero
caduti sulla liquidita' se fossero arrivati fin li'".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .config import (DEFAULT_MAX_DRAWDOWN, DEFAULT_MAX_PRICE,
                     DEFAULT_MIN_DOLLAR_VOLUME, DEFAULT_MIN_PRICE,
                     DEFAULT_MIN_SHARPE, DEFAULT_MIN_VOL, DEFAULT_RISK_FREE,
                     DEFAULT_YEARS_HISTORY, DEFAULT_YEARS_NO_NEG,
                     LIQUIDITY_LOOKBACK, MAX_ROLLING_YEARS, REJECT_LABELS,
                     REJECT_ORDER)

# Valore sentinella del notebook per uno Sharpe non calcolabile.
SHARPE_NON_CALCOLABILE = -999.0


@dataclass(frozen=True)
class Filtri:
    """
    Le soglie dello screener, tutte in un unico oggetto.

    I valori di partenza sono quelli della Cella 0 del notebook, cosi' chi
    esegue l'applicazione senza toccare nulla ottiene lo stesso universo.
    """

    anni_storico: int = DEFAULT_YEARS_HISTORY
    anni_perf_continua: int = DEFAULT_YEARS_NO_NEG
    max_drawdown: float = DEFAULT_MAX_DRAWDOWN
    min_sharpe: float = DEFAULT_MIN_SHARPE
    min_volatilita: float = DEFAULT_MIN_VOL
    prezzo_minimo: float = DEFAULT_MIN_PRICE
    prezzo_massimo: float = DEFAULT_MAX_PRICE
    dollar_volume_minimo: float = DEFAULT_MIN_DOLLAR_VOLUME
    tasso_privo_rischio: float = DEFAULT_RISK_FREE

    def come_dizionario(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Sharpe ratio
# --------------------------------------------------------------------------
def sharpe(metriche: pd.DataFrame, tasso_privo_rischio: float) -> pd.Series:
    """
    Sharpe ratio annualizzato di ciascun titolo, al tasso indicato.

    Si ricostruisce qui e non in archivio perche' il tasso privo di rischio
    e' un parametro dello studente: cambiarlo deve costare zero.

    I due casi degeneri del notebook - nessun rendimento disponibile, oppure
    deviazione standard nulla - restituiscono il sentinella -999, che fa
    fallire qualunque soglia ragionevole.
    """
    media = metriche["media_ann"].to_numpy(dtype=np.float64, copy=False)
    dev = metriche["dev_std_ann"].to_numpy(dtype=np.float64, copy=False)
    n_rend = metriche["n_rendimenti"].to_numpy(dtype=np.float64, copy=False)

    valido = (n_rend > 0) & np.isfinite(dev) & (dev != 0) & np.isfinite(media)
    with np.errstate(invalid="ignore", divide="ignore"):
        valori = np.where(valido, (media - tasso_privo_rischio) / dev,
                          SHARPE_NON_CALCOLABILE)
    return pd.Series(valori, index=metriche.index, name="sharpe")


# --------------------------------------------------------------------------
# Applicazione dei filtri
# --------------------------------------------------------------------------
def applica(metriche: pd.DataFrame, filtri: Filtri) -> pd.DataFrame:
    """
    Aggiunge alle metriche le colonne 'sharpe' ed 'esito'.

    'esito' vale 'PASSATO' per i titoli che superano tutti i filtri, oppure
    il codice del primo filtro che li ha esclusi.
    """
    df = metriche.copy()
    df["sharpe"] = sharpe(df, filtri.tasso_privo_rischio)

    n_oss = df["n_oss"].to_numpy(dtype=np.float64, copy=False)
    anni = df["anni_storico"].to_numpy(dtype=np.float64, copy=False)
    prezzo = df["prezzo_ultimo"].to_numpy(dtype=np.float64, copy=False)
    liquidita = df[f"dollar_volume_{LIQUIDITY_LOOKBACK}"].to_numpy(
        dtype=np.float64, copy=False)
    dd = df["max_drawdown"].to_numpy(dtype=np.float64, copy=False)
    sr = df["sharpe"].to_numpy(dtype=np.float64, copy=False)
    vol = df["volatilita_mensile"].to_numpy(dtype=np.float64, copy=False)

    finestra = int(np.clip(filtri.anni_perf_continua, 1, MAX_ROLLING_YEARS))
    roll = df[f"min_roll_{finestra}a"].to_numpy(dtype=np.float64, copy=False)

    # I confronti sono scritti come "condizione di SCARTO", nello stesso
    # ordine del notebook. Attenzione ai NaN: qui vanno gestiti a mano,
    # perche' "NaN < soglia" e' falso e lascerebbe passare il titolo.
    condizioni = [
        # Dati insufficienti o non misurabili.
        (n_oss <= 0) | ~np.isfinite(anni) | ~np.isfinite(prezzo),
        # Anzianita': la tolleranza di un decimo di anno assorbe i
        # disallineamenti fra calendari di borsa.
        anni < (filtri.anni_storico - 0.1),
        prezzo > filtri.prezzo_massimo,
        prezzo < filtri.prezzo_minimo,
        # Liquidita': NaN significa "storia piu' corta della finestra", e in
        # quel caso il notebook non applicava il filtro. Si fa lo stesso.
        np.isfinite(liquidita) & (liquidita < filtri.dollar_volume_minimo),
        # Performance continua: NaN significa "non c'e' abbastanza storia
        # per formare la finestra", e va scartato.
        ~np.isfinite(roll) | (roll < 0.0),
        ~np.isfinite(dd) | (dd < -filtri.max_drawdown),
        sr < filtri.min_sharpe,
        ~np.isfinite(vol) | (vol < filtri.min_volatilita),
    ]
    codici = ["DATI", "STORICO", "PREZZO_MAX", "PREZZO_MIN", "LIQUIDITA",
              "PERF_CONTINUA", "DRAWDOWN", "SHARPE", "VOLATILITA"]

    df["esito"] = np.select(condizioni, codici, default="PASSATO")
    return df


def riepilogo(risultato: pd.DataFrame) -> pd.DataFrame:
    """
    Quanti titoli sono caduti su ciascun filtro, nell'ordine di applicazione.

    E' la tabella piu' istruttiva dell'intera applicazione: dice a colpo
    d'occhio quale vincolo sta davvero decidendo la selezione. Nove volte su
    dieci e' l'anzianita' dello storico, non lo Sharpe.
    """
    conteggi = risultato["esito"].value_counts()
    totale = int(len(risultato))
    righe = []
    for codice in ["PASSATO"] + REJECT_ORDER:
        n = int(conteggi.get(codice, 0))
        righe.append({
            "Esito": REJECT_LABELS.get(codice, codice),
            "Titoli": n,
            "Quota": (n / totale) if totale else 0.0,
        })
    return pd.DataFrame(righe)


def promossi(risultato: pd.DataFrame) -> list[str]:
    """Elenco ordinato dei ticker che hanno superato tutti i filtri."""
    return sorted(risultato.loc[risultato["esito"] == "PASSATO", "ticker"].tolist())


def dettaglio_promossi(risultato: pd.DataFrame,
                       tasso_privo_rischio: float = 0.0) -> pd.DataFrame:
    """Tabella leggibile dei titoli promossi, ordinata per Sharpe."""
    colonne = ["ticker", "anni_storico", "prezzo_ultimo",
               f"dollar_volume_{LIQUIDITY_LOOKBACK}", "cagr",
               "volatilita_mensile", "dev_std_ann", "max_drawdown", "sharpe"]
    df = risultato.loc[risultato["esito"] == "PASSATO", colonne].copy()
    return df.sort_values("sharpe", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Sensibilita' di un singolo filtro
# --------------------------------------------------------------------------
def curva_sensibilita(metriche: pd.DataFrame, filtri: Filtri,
                      parametro: str, valori) -> pd.DataFrame:
    """
    Quanti titoli sopravvivono al variare di UNA soglia, tenendo ferme le altre.

    Serve a rispondere alla domanda che ogni studente si pone davanti a un
    universo vuoto: "quale filtro devo allentare, e di quanto?".
    """
    righe = []
    for v in valori:
        prova = Filtri(**{**filtri.come_dizionario(), parametro: v})
        esiti = applica(metriche, prova)["esito"]
        righe.append({"valore": v, "promossi": int((esiti == "PASSATO").sum())})
    return pd.DataFrame(righe)
