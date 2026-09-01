#!/usr/bin/env python
"""
Archivio finto, per provare l'applicazione senza chiave API.

Genera titoli sintetici con caratteristiche verosimili - chi e' quotato da
vent'anni e chi da due, chi scambia miliardi e chi niente, chi cresce
regolare e chi ha attraversato il 2008 perdendo l'ottanta per cento - e
scrive un archivio nella stessa identica forma di quello vero.

Serve a due cose: verificare che l'interfaccia funzioni prima di consumare
ventitremila chiamate API, e mostrare l'applicazione in aula anche quando la
rete non collabora.

    python scripts/archivio_demo.py --titoli 1200 --out data

I dati NON sono dati di mercato. Non ricavarne alcuna conclusione.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import datastore, metrics                       # noqa: E402
from src.config import BENCHMARK, SHARD_SIZE            # noqa: E402

# Momenti in cui il mercato e' sceso tutto insieme: servono a rendere
# realistiche le correlazioni, che altrimenti sarebbero quasi nulle e la
# mappa delle correlazioni non direbbe niente.
CRISI = [("2008-09-01", "2009-03-09", -0.0022),
         ("2020-02-19", "2020-03-23", -0.0180),
         ("2022-01-04", "2022-10-12", -0.0009)]


def genera(quanti: int, anni_max: int, seme: int,
           quota_regolari: float = 0.09) -> tuple[list, dict, list]:
    """
    Costruisce l'universo finto.

    Una quota di titoli - i "regolari" - viene generata apposta con deriva
    solida e volatilita' contenuta, il profilo del compounder che attraversa
    le crisi senza spezzarsi. Senza di loro nessun titolo supererebbe il
    filtro della performance continua, e la dimostrazione mostrerebbe uno
    screener che non seleziona mai niente: vero come comportamento del
    codice, inutile come dimostrazione.
    """
    rng = np.random.default_rng(seme)
    fine = pd.Timestamp.today().normalize()
    calendario = pd.bdate_range(end=fine, periods=int(anni_max * 252) + 400)

    # Un fattore di mercato comune a tutti i titoli, con dentro le tre crisi.
    mercato = rng.normal(0.00030, 0.0095, len(calendario))
    for inizio, termine, quanto in CRISI:
        dentro = (calendario >= inizio) & (calendario <= termine)
        mercato[dentro] += quanto

    anagrafica, serie, righe = [], {}, []

    for i in range(quanti):
        ticker = f"DEMO{i:04d}.US" if i else BENCHMARK
        regolare = False

        # Un titolo ogni duecentocinquanta esiste nel listino ma l'API non
        # restituisce nulla per lui. Non e' un dettaglio pittoresco: sul
        # mercato vero succede, e sono proprio questi titoli - presenti
        # nell'universo, assenti dal pannello prezzi - a rompere il codice
        # che dia per scontato che le due cose coincidano.
        if i and i % 250 == 0:
            righe.append(metrics.riga_vuota(ticker))
            anagrafica.append({
                "ticker": ticker, "code": ticker.split(".")[0],
                "name": f"Societa' senza quotazioni {i}",
                "type": "Common Stock", "exchange": "US", "currency": "USD",
            })
            continue

        if ticker == BENCHMARK:
            anni, beta, alfa, idio, volume = anni_max, 1.0, 0.0, 0.0025, 4e8
            base = 120.0
        elif rng.random() < quota_regolari:
            # Il compounder: cresce meno del mercato nelle euforie e scende
            # molto meno nelle crisi, ma non smette mai di salire.
            regolare = True
            anni = float(rng.uniform(20.5, anni_max))
            beta = float(rng.uniform(0.45, 0.85))
            alfa = float(rng.uniform(0.00035, 0.00070))
            idio = float(rng.uniform(0.0060, 0.0105))
            volume = float(10 ** rng.uniform(7.6, 9.0))
            base = float(rng.uniform(15.0, 250.0))
        else:
            anni = float(rng.choice([1, 3, 7, 12, 18, 21, 24],
                                    p=[.10, .14, .16, .15, .13, .20, .12]))
            beta = float(rng.normal(1.0, 0.45))
            alfa = float(rng.normal(0.00005, 0.00028))
            idio = float(rng.choice([0.006, 0.011, 0.019, 0.032]))
            volume = float(10 ** rng.uniform(4.0, 9.0))
            base = float(rng.choice([1.5, 8.0, 45.0, 180.0, 900.0]))

        n = min(len(calendario), max(60, int(anni * 252)))
        date = calendario[-n:]
        shock = alfa + beta * mercato[-n:] + rng.normal(0, idio, n)
        prezzi = base * np.exp(np.cumsum(shock))
        pezzi = np.abs(rng.normal(volume / max(prezzi.mean(), 1e-6),
                                  volume / max(prezzi.mean(), 1e-6) * 0.3, n))

        date_np = date.to_numpy(dtype="datetime64[D]")
        prezzi = prezzi.astype(np.float32)
        serie[ticker] = (date_np, prezzi)

        riga = metrics.misura(ticker, date_np, prezzi, pezzi.astype(np.float32))
        righe.append(riga if riga is not None else metrics.riga_vuota(ticker))

        anagrafica.append({
            "ticker": ticker, "code": ticker.split(".")[0],
            "name": "Benchmark sintetico" if ticker == BENCHMARK
                    else (f"Compounder dimostrativo {i}" if regolare
                          else f"Societa' dimostrativa {i}"),
            "type": "ETF" if ticker == BENCHMARK else "Common Stock",
            "exchange": "US", "currency": "USD",
        })

    return anagrafica, serie, righe


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--titoli", type=int, default=1200)
    p.add_argument("--anni", type=int, default=22)
    p.add_argument("--out", default="data")
    p.add_argument("--seme", type=int, default=42)
    p.add_argument("--fetta", type=int, default=SHARD_SIZE)
    args = p.parse_args()

    destinazione = Path(args.out)
    print(f"Genero {args.titoli} titoli sintetici su {args.anni} anni...")
    anagrafica, serie, righe = genera(args.titoli, args.anni, args.seme)

    tickers = [r["ticker"] for r in anagrafica]
    mappa = datastore.assegna_fette(tickers, args.fetta)
    n_fette = max(mappa.values()) + 1

    per_fetta: dict[int, dict] = {i: {} for i in range(n_fette)}
    for t, dati in serie.items():
        per_fetta[mappa[t]][t] = dati

    totale_colonne = 0
    for indice in range(n_fette):
        totale_colonne += datastore.scrivi_fetta(per_fetta[indice], indice,
                                                 destinazione)
        print(f"  fetta {indice:02d} scritta")

    # Stesso riordino dell'archivio vero: i titoli con la storia piu' lunga
    # nelle prime fette, cosi' l'applicazione ne scarica poche.
    mappa = datastore.riordina_per_anzianita(pd.DataFrame(righe), mappa,
                                             destinazione, args.fetta)
    n_fette = max(mappa.values()) + 1 if mappa else 0

    tabella = datastore.scrivi_metriche(righe, mappa, destinazione)
    datastore.scrivi_universo(anagrafica, destinazione)

    oggi = dt.date.today()
    datastore.scrivi_meta({
        "costruito_il": dt.datetime.now(dt.timezone.utc).isoformat(),
        "universo": "DEMO",
        "universo_etichetta": "ARCHIVIO DIMOSTRATIVO - dati sintetici",
        "borse": ["DEMO"], "benchmark": BENCHMARK,
        "anni_storico": 20,
        "data_inizio": (oggi - dt.timedelta(days=args.anni * 365)).isoformat(),
        "data_fine": oggi.isoformat(),
        "n_universo": len(tickers), "n_misurati": len(tickers),
        "n_con_prezzi": totale_colonne,
        "n_con_storico_pieno": int((tabella["anni_storico"] >= 19.9).sum()),
        "n_fette": n_fette, "dimensione_fetta": args.fetta,
        "download_secondi": 0.0,
        "esiti_download": {"OK": len(tickers)},
        "dimostrativo": True,
    }, destinazione)

    peso = sum(f.stat().st_size for f in destinazione.rglob("*") if f.is_file())
    print(f"\nArchivio dimostrativo pronto in '{destinazione}'")
    print(f"  {len(tickers)} titoli, {totale_colonne} serie, "
          f"{peso / 1024 ** 2:.1f} MB")
    print(f"  con almeno 20 anni di storia: "
          f"{int((tabella['anni_storico'] >= 19.9).sum())}")
    print("\n  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
