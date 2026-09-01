#!/usr/bin/env python
"""
Costruzione dell'archivio dati.

Scarica l'intero universo dei titoli, ne misura ogni singolo elemento e
salva il risultato in formato Parquet. E' l'unica parte lenta del sistema,
e gira una volta ogni sei mesi su una macchina di GitHub - non davanti allo
studente.

Uso tipico (GitHub Actions):
    python scripts/build_dataset.py --universo USA --anni 20 --out dist

Uso in aula, per farsi un archivio piccolo e provare il giro completo:
    python scripts/build_dataset.py --limite 300 --out data

La chiave API si legge dalla variabile d'ambiente EODHD_API_KEY.

Il flusso e' pensato per non accumulare memoria: si lavora una fetta per
volta, e di ogni titolo si tiene soltanto la riga di misure (una trentina di
numeri) piu' la serie dei prezzi, che viene scritta su disco e liberata alla
fine della fetta. Cosi' il picco di memoria resta di poche centinaia di
megabyte anche su un universo di ventitremila titoli.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import datastore, eodhd, metrics                       # noqa: E402
from src.config import (ARCHIVE_MAX_AGE_DAYS, BENCHMARK,        # noqa: E402
                        DEFAULT_YEARS_HISTORY, DOWNLOAD_RATE_PER_MIN,
                        DOWNLOAD_WORKERS, SHARD_SIZE, UNIVERSE_CHOICES)


def log(messaggio: str = "") -> None:
    """Stampa con l'ora: nei log di GitHub Actions serve a capire i tempi."""
    if messaggio:
        print(f"[{dt.datetime.now():%H:%M:%S}] {messaggio}", flush=True)
    else:
        print(flush=True)


def durata(secondi: float) -> str:
    minuti, sec = divmod(int(secondi), 60)
    ore, minuti = divmod(minuti, 60)
    if ore:
        return f"{ore}h {minuti:02d}m {sec:02d}s"
    if minuti:
        return f"{minuti}m {sec:02d}s"
    return f"{sec}s"


def argomenti() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Costruisce l'archivio dello Screener Markowitz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--universo", choices=sorted(UNIVERSE_CHOICES),
                   default="USA", help="quali mercati includere")
    p.add_argument("--anni", type=int, default=DEFAULT_YEARS_HISTORY,
                   help="anni di storico da scaricare")
    p.add_argument("--out", default="data",
                   help="cartella di destinazione dell'archivio")
    p.add_argument("--limite", type=int, default=0,
                   help="scarica solo i primi N titoli (0 = tutti). Per le prove")
    p.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS,
                   help="download in parallelo")
    p.add_argument("--ritmo", type=int, default=DOWNLOAD_RATE_PER_MIN,
                   help="chiamate API al minuto")
    p.add_argument("--fetta", type=int, default=SHARD_SIZE,
                   help="titoli per fetta del pannello prezzi")
    p.add_argument("--se-piu-vecchio-di", type=int, default=0, dest="scadenza",
                   help="non fa nulla se l'archivio esistente e' piu' giovane "
                        "di N giorni (0 = ricostruisci sempre)")
    p.add_argument("--forza", action="store_true",
                   help="ricostruisce anche se l'archivio e' recente")
    return p.parse_args()


def segnala(destinazione: Path, esito: str) -> None:
    """
    Lascia scritto se si e' costruito o no.

    Serve al workflow di GitHub, che deve sapere se pubblicare una nuova
    release oppure non fare nulla. Un file di quattro parole e' piu' robusto
    di qualunque analisi dei log.
    """
    destinazione.mkdir(parents=True, exist_ok=True)
    (destinazione / "esito.txt").write_text(esito, encoding="utf-8")


def main() -> int:
    args = argomenti()
    destinazione = Path(args.out)

    log("=" * 66)
    log("Archivio Screener Markowitz - Scuola di Finanza Operativa")
    log("=" * 66)

    # ---- Serve davvero ricostruire? -------------------------------------
    if args.scadenza > 0 and not args.forza:
        meta = datastore.leggi_meta(destinazione)
        eta = datastore.eta_giorni(meta)
        if eta is not None and eta < args.scadenza:
            log(f"L'archivio ha {eta:.0f} giorni, la soglia e' {args.scadenza}.")
            log("Non c'e' nulla da fare. Esco senza consumare chiamate API.")
            segnala(destinazione, "saltato")
            return 0
    segnala(destinazione, "in-corso")

    # ---- Chiave ----------------------------------------------------------
    token = eodhd.get_api_key()
    if not token:
        log("ERRORE: chiave API EODHD non trovata.")
        log("")
        log("Ho cercato qui:")
        for riga in eodhd.diagnostica_chiave():
            log(f"  - {riga}")
        log("")
        log("Su GitHub: Settings -> Secrets and variables -> Actions ->")
        log("scheda SECRETS (non Variables) -> New repository secret.")
        log("Nome esatto: EODHD_API_KEY. Valore: la chiave nuda, senza virgolette.")
        log("In locale:   export EODHD_API_KEY=\"...\"")
        return 1
    log(f"Chiave API: {eodhd.mask_key(token)} ({len(token)} caratteri)")

    try:
        info = eodhd.fetch_user_info(token)
        log(f"Piano: {info.get('subscriptionType', 'n.d.')} | "
            f"chiamate usate oggi: {info.get('apiRequests', '?')} "
            f"su {info.get('dailyRateLimit', '?')}")
    except Exception as exc:
        log(f"Nota: non ho potuto leggere lo stato dell'abbonamento ({exc}).")

    # ---- Universo --------------------------------------------------------
    etichetta, configurazione = UNIVERSE_CHOICES[args.universo]
    log("")
    log(f"Universo richiesto: {etichetta}")
    log(f"Borse: {', '.join(configurazione)}")

    avvio = time.perf_counter()
    universo = eodhd.build_universe(
        configurazione, token, benchmark=BENCHMARK,
        on_progress=lambda borsa, n: log(f"  {borsa}: {n:,} simboli"),
    )
    if args.limite > 0:
        tenuti = {r["ticker"] for r in universo[:args.limite]}
        tenuti.add(BENCHMARK)
        universo = [r for r in universo if r["ticker"] in tenuti]
        log(f"  (limite di prova attivo: {len(universo)} titoli)")

    tickers = [r["ticker"] for r in universo]
    log(f"Universo completo: {len(tickers):,} titoli")

    if not tickers:
        log("ERRORE: universo vuoto. Controlla la copertura del piano EODHD.")
        return 1

    # ---- Finestra temporale ---------------------------------------------
    oggi = dt.date.today()
    # Un margine di sei mesi oltre gli anni richiesti: serve a non tagliare
    # via i titoli quotati proprio a ridosso del limite.
    inizio = oggi - dt.timedelta(days=int(args.anni * 365.25) + 180)
    log(f"Finestra: {inizio.isoformat()} -> {oggi.isoformat()} "
        f"({args.anni} anni + margine)")

    attesa = eodhd.stima_durata(len(tickers), args.ritmo)
    log(f"A {args.ritmo} chiamate/minuto con {args.workers} thread: "
        f"circa {attesa:.0f} minuti di download.")

    # ---- Fette -----------------------------------------------------------
    mappa_fette = datastore.assegna_fette(tickers, args.fetta)
    n_fette = max(mappa_fette.values()) + 1
    log(f"Il pannello prezzi sara' diviso in {n_fette} fette da {args.fetta}.")
    log("")

    per_fetta: dict[int, list[str]] = {}
    for t, f in mappa_fette.items():
        per_fetta.setdefault(f, []).append(t)

    # ---- Download e misura ----------------------------------------------
    righe_metriche: list[dict] = []
    stato = {"OK": 0, "VUOTO": 0, "ASSENTE": 0, "ERRORE": 0, "CORTO": 0}
    n_colonne_totali = 0
    fatti = 0
    avvio_download = time.perf_counter()

    for indice in range(n_fette):
        elenco = sorted(per_fetta.get(indice, []))
        if not elenco:
            datastore.scrivi_fetta({}, indice, destinazione)
            continue

        serie: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for grezza in eodhd.download_series(
                elenco, token, inizio.isoformat(), oggi.isoformat(),
                workers=args.workers, per_minute=args.ritmo):

            fatti += 1
            if grezza.stato != "OK":
                chiave = ("ERRORE" if grezza.stato.startswith("ERRORE")
                          else grezza.stato)
                stato[chiave] = stato.get(chiave, 0) + 1
                righe_metriche.append(metrics.riga_vuota(grezza.ticker))
            else:
                riga = metrics.misura(grezza.ticker, grezza.date,
                                      grezza.close, grezza.volume)
                if riga is None:
                    # Meno di un anno di quotazioni: si registra comunque,
                    # perche' l'universo dello screener deve restare completo.
                    stato["CORTO"] += 1
                    righe_metriche.append(metrics.riga_vuota(grezza.ticker))
                else:
                    stato["OK"] += 1
                    righe_metriche.append(riga)
                # La serie serve al pannello anche se corta: sara' lo
                # screener a decidere se il titolo e' utilizzabile.
                serie[grezza.ticker] = (grezza.date, grezza.close)

            if fatti % 500 == 0:
                trascorso = time.perf_counter() - avvio_download
                ritmo = fatti / trascorso * 60.0
                rimasti = len(tickers) - fatti
                log(f"  {fatti:>6,}/{len(tickers):,} "
                    f"({100 * fatti / len(tickers):4.1f}%) | "
                    f"{ritmo:.0f}/min | mancano {durata(rimasti / max(ritmo, 1) * 60)}")

        n_colonne = datastore.scrivi_fetta(serie, indice, destinazione)
        n_colonne_totali += n_colonne
        peso = datastore.percorso_fetta(indice, destinazione).stat().st_size
        log(f"  fetta {indice:02d}: {n_colonne:,} serie salvate "
            f"({peso / 1024 ** 2:.1f} MB)")
        serie.clear()

    secondi_download = time.perf_counter() - avvio_download
    log("")
    log(f"Download concluso in {durata(secondi_download)}")
    log(f"  scaricati e misurati : {stato['OK']:,}")
    log(f"  storia troppo corta  : {stato['CORTO']:,}")
    log(f"  senza dati           : {stato['VUOTO']:,}")
    log(f"  non trovati (404)    : {stato['ASSENTE']:,}")
    log(f"  errori               : {stato['ERRORE']:,}")

    # ---- Riordino delle fette --------------------------------------------
    # Fino a qui le fette sono in ordine alfabetico, l'unico noto prima di
    # scaricare. Ora che si sa quanto e' lungo lo storico di ciascun titolo,
    # si rimescolano mettendo davanti i piu' anziani: sono gli unici che la
    # simulazione potra' usare, e cosi' l'applicazione ne scarichera' due o
    # tre invece di tutte.
    log("")
    log("Riordino le fette per anzianita' dello storico...")
    mappa_finale = datastore.riordina_per_anzianita(
        pd.DataFrame(righe_metriche), mappa_fette, destinazione, args.fetta,
        on_progress=lambda fatte, totali: log(f"  fetta {fatte}/{totali}"),
    )
    n_fette = max(mappa_finale.values()) + 1 if mappa_finale else 0

    # ---- Tabelle finali --------------------------------------------------
    log("")
    log("Scrivo le tabelle...")
    tabella = datastore.scrivi_metriche(righe_metriche, mappa_finale, destinazione)
    datastore.scrivi_universo(universo, destinazione)

    con_venti_anni = int((tabella["anni_storico"] >= args.anni - 0.1).sum())

    meta = datastore.scrivi_meta({
        "costruito_il": dt.datetime.now(dt.timezone.utc).isoformat(),
        "universo": args.universo,
        "universo_etichetta": etichetta,
        "borse": list(configurazione),
        "benchmark": BENCHMARK,
        "anni_storico": args.anni,
        "data_inizio": inizio.isoformat(),
        "data_fine": oggi.isoformat(),
        "n_universo": len(tickers),
        "n_misurati": stato["OK"],
        "n_con_prezzi": n_colonne_totali,
        "n_con_storico_pieno": con_venti_anni,
        "n_fette": n_fette,
        "dimensione_fetta": args.fetta,
        "download_secondi": round(secondi_download, 1),
        "esiti_download": stato,
    }, destinazione)

    segnala(destinazione, "costruito")

    peso = sum(p.stat().st_size for p in destinazione.rglob("*") if p.is_file())
    log("")
    log("=" * 66)
    log(f"Archivio pronto in '{destinazione}'")
    log(f"  titoli nell'universo          : {len(tickers):,}")
    log(f"  titoli con serie storiche     : {n_colonne_totali:,}")
    log(f"  titoli con {args.anni} anni pieni       : {con_venti_anni:,}")
    log(f"  peso totale                   : {peso / 1024 ** 2:.0f} MB")
    log(f"  tempo totale                  : {durata(time.perf_counter() - avvio)}")
    log("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
