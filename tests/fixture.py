"""
Materiale sintetico per i test. NON e' un archivio utilizzabile.

Qui dentro nascono gli unici dati finti dell'intero progetto, e nascono con
tre lucchetti che rendono impossibile scambiarli per dati veri:

  1. vivono solo in una cartella temporanea, creata e distrutta dal test.
     Non toccano mai 'data/', che e' l'unico posto da cui l'applicazione
     legge;
  2. il loro meta.json dichiara "origine": "test", e l'applicazione si
     RIFIUTA di aprire qualunque archivio che non dichiari "eodhd";
  3. stanno in tests/, non in scripts/: non c'e' alcun comando che un utente
     possa lanciare per generarli.

Servono a una cosa sola, ma importante: provare la catena completa -
scrittura dell'archivio, riordino delle fette, lettura, screening,
simulazione, report - senza consumare ventiquattromila chiamate API. E' il
controllo che intercetta gli errori strutturali prima che costino
venticinque minuti di download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import datastore, metrics
from src.config import BENCHMARK

# Momenti in cui il mercato scende tutto insieme. Servono a rendere
# realistiche le correlazioni: senza, la matrice sarebbe quasi diagonale e
# non metterebbe alla prova il codice che la calcola e la disegna.
CRISI = [("2008-09-01", "2009-03-09", -0.0022),
         ("2020-02-19", "2020-03-23", -0.0180),
         ("2022-01-04", "2022-10-12", -0.0009)]

# Un titolo ogni quanti nasce senza quotazioni. Non e' folklore: sul listino
# vero succede, ed e' esattamente il caso - presente nell'universo, assente
# dal pannello prezzi - che ha fatto fallire la prima costruzione completa.
UNO_SENZA_DATI_OGNI = 250

# Quota di titoli generati con deriva solida e volatilita' contenuta. Senza
# di loro nessun titolo supererebbe il filtro della performance continua e la
# prova si fermerebbe al primo passo.
QUOTA_REGOLARI = 0.09


def costruisci(cartella: Path | str, quanti: int = 600, anni: int = 22,
               seme: int = 42, dimensione_fetta: int = 250) -> dict:
    """
    Scrive un archivio di prova nella cartella indicata.

    La cartella deve essere temporanea: questa funzione non ha alcun
    meccanismo per impedirti di puntarla su 'data/', ma l'applicazione
    rifiutera' comunque di aprire quello che ne esce.
    """
    cartella = Path(cartella)
    rng = np.random.default_rng(seme)
    fine = pd.Timestamp.today().normalize()
    calendario = pd.bdate_range(end=fine, periods=int(anni * 252) + 400)

    mercato = rng.normal(0.00030, 0.0095, len(calendario))
    for inizio, termine, quanto in CRISI:
        dentro = (calendario >= inizio) & (calendario <= termine)
        mercato[dentro] += quanto

    anagrafica: list[dict] = []
    serie: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    righe: list[dict] = []

    for i in range(quanti):
        ticker = f"PROVA{i:04d}.US" if i else BENCHMARK

        if i and i % UNO_SENZA_DATI_OGNI == 0:
            # Esiste nel listino, l'API non restituisce nulla.
            righe.append(metrics.riga_vuota(ticker))
            anagrafica.append(_anagrafe(ticker, i, "Senza quotazioni"))
            continue

        if ticker == BENCHMARK:
            anni_t, beta, alfa, idio, volume, base = anni, 1.0, 0.0, 0.0025, 4e8, 120.0
        elif rng.random() < QUOTA_REGOLARI:
            anni_t = float(rng.uniform(20.5, anni))
            beta = float(rng.uniform(0.45, 0.85))
            alfa = float(rng.uniform(0.00035, 0.00070))
            idio = float(rng.uniform(0.0060, 0.0105))
            volume = float(10 ** rng.uniform(7.6, 9.0))
            base = float(rng.uniform(15.0, 250.0))
        else:
            anni_t = float(rng.choice([1, 3, 7, 12, 18, 21, 24],
                                      p=[.10, .14, .16, .15, .13, .20, .12]))
            beta = float(rng.normal(1.0, 0.45))
            alfa = float(rng.normal(0.00005, 0.00028))
            idio = float(rng.choice([0.006, 0.011, 0.019, 0.032]))
            volume = float(10 ** rng.uniform(4.0, 9.0))
            base = float(rng.choice([1.5, 8.0, 45.0, 180.0, 900.0]))

        n = min(len(calendario), max(60, int(anni_t * 252)))
        date = calendario[-n:]
        shock = alfa + beta * mercato[-n:] + rng.normal(0, idio, n)
        prezzi = (base * np.exp(np.cumsum(shock))).astype(np.float32)
        pezzi = np.abs(rng.normal(volume / max(float(prezzi.mean()), 1e-6),
                                  volume / max(float(prezzi.mean()), 1e-6) * 0.3, n))

        date_np = date.to_numpy(dtype="datetime64[D]")
        serie[ticker] = (date_np, prezzi)
        riga = metrics.misura(ticker, date_np, prezzi, pezzi.astype(np.float32))
        righe.append(riga if riga is not None else metrics.riga_vuota(ticker))
        anagrafica.append(_anagrafe(ticker, i, "Titolo di prova"))

    tickers = [r["ticker"] for r in anagrafica]
    mappa = datastore.assegna_fette(tickers, dimensione_fetta)
    n_fette = max(mappa.values()) + 1

    per_fetta: dict[int, dict] = {i: {} for i in range(n_fette)}
    for t, dati in serie.items():
        per_fetta[mappa[t]][t] = dati

    colonne = 0
    for indice in range(n_fette):
        colonne += datastore.scrivi_fetta(per_fetta[indice], indice, cartella)

    # Stesso riordino dell'archivio vero: e' proprio quello che va provato.
    mappa = datastore.riordina_per_anzianita(pd.DataFrame(righe), mappa,
                                             cartella, dimensione_fetta)
    tabella = datastore.scrivi_metriche(righe, mappa, cartella)
    datastore.scrivi_universo(anagrafica, cartella)

    meta = datastore.scrivi_meta({
        # Il lucchetto. L'applicazione apre solo gli archivi con
        # "origine": "eodhd", cioe' quelli usciti da build_dataset.py.
        "origine": "test",
        "universo": "PROVA",
        "universo_etichetta": "DATI SINTETICI - solo per i test",
        "borse": ["PROVA"],
        "benchmark": BENCHMARK,
        "anni_storico": 20,
        "n_universo": len(tickers),
        "n_misurati": int((tabella["n_oss"] > 0).sum()),
        "n_con_prezzi": colonne,
        "n_con_storico_pieno": int((tabella["anni_storico"] >= 19.9).sum()),
        "n_fette": max(mappa.values()) + 1 if mappa else 0,
        "dimensione_fetta": dimensione_fetta,
    }, cartella)

    return meta


def _anagrafe(ticker: str, i: int, descrizione: str) -> dict:
    return {
        "ticker": ticker,
        "code": ticker.split(".")[0],
        "name": f"{descrizione} {i}",
        "type": "ETF" if ticker == BENCHMARK else "Common Stock",
        "exchange": "PROVA",
        "currency": "USD",
    }
