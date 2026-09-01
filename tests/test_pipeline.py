"""
Prova d'insieme: dall'archivio al report, senza interfaccia.

Percorre esattamente la strada che percorre l'applicazione - legge le
misure, filtra, carica le serie dei promossi, simula, ricostruisce le curve,
calcola le statistiche rolling, disegna i grafici e compone il report HTML -
e verifica che ogni passaggio produca qualcosa di sensato.

Richiede un archivio in 'data'. Se non c'e', lo si crea in un attimo:
    python scripts/archivio_demo.py --titoli 800

Si esegue con:
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

from src import charts, datastore, portfolio, report, screener      # noqa: E402
from src.config import (BENCHMARK, FRONTIER_MAX_POINTS,             # noqa: E402
                        HEATMAP_MAX_TICKERS, HEATMAP_TOP_PORTFOLIOS,
                        MIN_TICKERS_FOR_MAX_RETURN, ROLLING_WINDOWS)


def cronometro(etichetta: str, funzione, *args, **kwargs):
    avvio = time.perf_counter()
    esito = funzione(*args, **kwargs)
    print(f"  {etichetta:<44} {time.perf_counter() - avvio:7.3f}s")
    return esito


def main() -> int:
    if not datastore.archivio_presente():
        print("Manca l'archivio. Esegui prima:")
        print("  python scripts/archivio_demo.py --titoli 800")
        return 1

    meta = datastore.leggi_meta() or {}
    print(f"\nArchivio: {meta.get('universo_etichetta')} "
          f"({meta.get('n_universo')} titoli)")
    if meta.get("dimostrativo"):
        print("ATTENZIONE: dati sintetici, nessun valore informativo.\n")

    # ---- Screening -------------------------------------------------------
    print("Screening")
    metriche = cronometro("lettura della tabella delle misure",
                          datastore.leggi_metriche)
    assert not metriche.empty
    assert "fetta" in metriche.columns

    # I titoli senza serie di prezzi devono restare nell'universo con
    # l'etichetta "nessuna fetta". E' il caso che ha fatto fallire la prima
    # costruzione completa: due titoli su ventitremila esistevano nel listino
    # ma l'API non restituiva quotazioni, e il codice dava per scontato che
    # ogni riga delle misure avesse un pannello dietro.
    senza = metriche[metriche["fetta"] == datastore.SENZA_FETTA]
    assert metriche["fetta"].notna().all(), "nessuna fetta puo' essere NaN"
    assert (senza["n_oss"] == 0).all(), \
        "un titolo senza fetta non puo' avere osservazioni"
    print(f"  -> {len(senza)} titoli senza serie di prezzi, correttamente "
          f"marcati con fetta {datastore.SENZA_FETTA}")
    if len(senza):
        # Chiederli non deve rompere nulla: vanno semplicemente ignorati.
        mappa_prova = datastore.mappa_fette_da_metriche(metriche)
        vuoto = datastore.leggi_prezzi(senza["ticker"].tolist()[:3],
                                       mappa_prova)
        assert vuoto.empty, "un titolo senza fetta non deve restituire prezzi"
        datastore.assicura_fette([datastore.SENZA_FETTA])   # non deve sollevare

    # Si parte dalle soglie predefinite - le stesse della Cella 0 del
    # notebook - e se l'archivio in uso e' troppo magro si allarga.
    # Attenzione: allargare NON significa accorciare la finestra della
    # performance continua. Una finestra piu' corta e' piu' difficile da
    # tenere positiva, non piu' facile: tre anni che contengono solo il 2008
    # chiudono in perdita, cinque anni che contengono anche il recupero no.
    tentativi = [
        screener.Filtri(),
        screener.Filtri(min_sharpe=0.0, max_drawdown=0.85,
                        dollar_volume_minimo=1e6, prezzo_minimo=1.0),
        screener.Filtri(anni_perf_continua=8, min_sharpe=-1.0,
                        max_drawdown=0.99, dollar_volume_minimo=0.0,
                        prezzo_minimo=0.0, prezzo_massimo=1e6),
    ]
    for numero, filtri in enumerate(tentativi, start=1):
        risultato = cronometro(
            f"applicazione dei filtri all'universo intero ({numero})",
            screener.applica, metriche, filtri)
        riepilogo = screener.riepilogo(risultato)
        promossi = screener.promossi(risultato)
        print(f"  -> {len(promossi)} promossi su {len(metriche)}")
        assert riepilogo["Titoli"].sum() == len(metriche), \
            "il riepilogo deve contare tutti i titoli, una volta sola"
        if len(promossi) >= 6:
            break
    assert len(promossi) >= 6, "servono almeno sei titoli per il resto della prova"

    # ---- Prezzi ----------------------------------------------------------
    print("\nSerie storiche")
    mappa = datastore.mappa_fette_da_metriche(metriche)
    elenco = sorted(set(promossi) | {BENCHMARK})
    prezzi = cronometro(f"lettura di {len(elenco)} serie dalle fette",
                        datastore.leggi_prezzi, elenco, mappa)
    assert not prezzi.empty
    print(f"  -> {prezzi.shape[1]} colonne, {prezzi.shape[0]} giorni, "
          f"dal {prezzi.index[0].date()} al {prezzi.index[-1].date()}")

    universo = cronometro("preparazione dei rendimenti",
                          portfolio.prepara_universo, prezzi, BENCHMARK, 20)
    print(f"  -> {len(universo.tickers)} utilizzabili, "
          f"{len(universo.scartati)} esclusi perche' troppo giovani")
    assert len(universo.tickers) >= 5

    # ---- Simulazione -----------------------------------------------------
    print("\nSimulazione")
    mu, cov = cronometro("medie e covarianze", portfolio.matrici, universo)
    assert mu.shape[0] == len(universo.tickers)
    assert cov.shape == (len(universo.tickers), len(universo.tickers))

    sim = cronometro("Monte Carlo su 50.000 portafogli", portfolio.simula,
                     mu, cov, universo.tickers, 5, 10, 50_000, 0.0, 42)
    assert len(sim.portafogli) == 50_000
    assert np.isfinite(sim.portafogli["Sharpe_Ratio"]).all()

    chiave = portfolio.portafogli_chiave(sim, MIN_TICKERS_FOR_MAX_RETURN)
    assert len(chiave) >= 2
    for nome, dati in chiave.items():
        print(f"  {nome:<22} rend {dati['rendimento']:>7.2%} | "
              f"vol {dati['volatilita']:>6.2%} | "
              f"Sharpe {dati['sharpe']:>6.3f} | {dati['num_tickers']} titoli")

    campione = portfolio.campione_frontiera(sim, FRONTIER_MAX_POINTS)
    assert len(campione) == min(FRONTIER_MAX_POINTS, len(sim.portafogli))

    # ---- Serie dei portafogli -------------------------------------------
    print("\nRicostruzione delle serie")
    serie, curve, cadute = {}, {}, {}
    for metodo in ("NONE", "BUYHOLD", "QUARTERLY", "SEMIANNUALLY", "ANNUALLY"):
        s = portfolio.serie_portafoglio(
            universo.rendimenti, chiave["Max Sharpe"]["tickers"], metodo)
        assert not s.empty and np.isfinite(s.to_numpy()).all(), metodo
        m = portfolio.misure(s)
        print(f"  {metodo:<14} CAGR {m['cagr']:>7.2%} | "
              f"drawdown {m['max_drawdown']:>7.2%} | {m['giorni']} giorni")

    for nome, dati in chiave.items():
        s = portfolio.serie_portafoglio(universo.rendimenti, dati["tickers"],
                                        "NONE")
        serie[nome] = s
        curve[nome] = portfolio.equity(s)
        cadute[nome] = portfolio.drawdown(s)
    if universo.benchmark:
        s = universo.rendimenti[universo.benchmark].dropna()
        serie["SPY (benchmark)"] = s
        curve["SPY (benchmark)"] = portfolio.equity(s)
        cadute["SPY (benchmark)"] = portfolio.drawdown(s)

    misure = pd.DataFrame([
        {"Strategia": nome, "Dal": m["inizio"].date(), "Al": m["fine"].date(),
         "Anni": m["anni"], "CAGR": m["cagr"], "Volatilita": m["volatilita"],
         "Sharpe": m["sharpe"], "Drawdown max": m["max_drawdown"],
         "Rendimento totale": m["rendimento_totale"]}
        for nome, s in serie.items() if (m := portfolio.misure(s))
    ])
    assert len(misure) == len(serie)

    rolling = {}
    avvio = time.perf_counter()
    for nome, s in serie.items():
        rolling[nome] = portfolio.statistiche_rolling(s, ROLLING_WINDOWS)
    print(f"  statistiche rolling per {len(serie)} strategie: "
          f"{time.perf_counter() - avvio:.3f}s")

    frequenti = portfolio.titoli_piu_frequenti(sim, HEATMAP_TOP_PORTFOLIOS,
                                               HEATMAP_MAX_TICKERS)
    matrice = portfolio.correlazioni(universo.rendimenti, frequenti)
    print(f"  correlazioni fra i {len(matrice)} titoli piu' ricorrenti")

    # ---- Grafici ---------------------------------------------------------
    print("\nGrafici")
    figure = {
        "frontiera": charts.frontiera(campione, chiave, "prova"),
        "equity": charts.equity(curve, "prova", True),
        "underwater": charts.underwater(cadute),
        "correlazioni": charts.correlazioni(matrice),
        "imbuto": charts.imbuto_filtri(riepilogo),
        "nuvola": charts.nuvola_universo(risultato),
        "contributi": charts.contributo_titoli(
            universo.rendimenti, chiave["Max Sharpe"]["tickers"], "Max Sharpe"),
    }
    for nome, fig in figure.items():
        assert fig is not None and len(fig.data) > 0, nome
    print(f"  {len(figure)} grafici costruiti, tutti con almeno una serie")

    # ---- Report ----------------------------------------------------------
    print("\nReport")
    html = cronometro("composizione dell'HTML", report.componi,
                      meta=meta, filtri=filtri,
                      parametri={"ribilanciamento": "NONE",
                                 "n_simulazioni": 50_000, "seme": 42,
                                 "min_titoli": 5, "max_titoli": 10},
                      riepilogo=riepilogo, chiave=chiave, misure=misure,
                      migliori=portfolio.migliori(sim, 15), rolling=rolling,
                      figure=figure)
    assert html.startswith("<!DOCTYPE html>")
    assert "plotly" in html.lower()
    assert html.count("<h2>") >= 8
    destinazione = RADICE / "report_di_prova.html"
    destinazione.write_text(html, encoding="utf-8")
    print(f"  -> {len(html) / 1024 ** 2:.1f} MB scritti in {destinazione.name}")

    print("\nTutta la catena funziona.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
