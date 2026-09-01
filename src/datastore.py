"""
L'archivio: come si scrive, come si legge, come si aggiorna.

L'archivio e' una fotografia dell'intero mercato scattata una volta ogni sei
mesi. Contiene le misure di TUTTI i titoli dell'universo - nessuno escluso,
nessun pre-filtro - e le serie storiche dei prezzi divise in fette.

Perche' a fette. Un unico file parquet con ventitremila colonne funziona,
ma il suo indice interno diventa enorme e va letto per intero anche solo per
sapere che cosa c'e' dentro. Diviso in sedici fette da millecinquecento
titoli, invece, l'applicazione apre soltanto le fette che contengono i
titoli sopravvissuti ai filtri: nella pratica due o tre, qualche decina di
megabyte invece di duecento.

Perche' non si aggiorna in modo incrementale. Perche' l'adjusted_close
viene ricalcolato ALL'INDIETRO su tutta la serie ogni volta che c'e' un
frazionamento o uno stacco di dividendo. Accodare gli ultimi sei mesi a una
serie vecchia produrrebbe un gradino artificiale nel punto di giunzione, e
quel gradino finirebbe dritto nel calcolo del drawdown. Si ricostruisce da
zero, e costa mezz'ora di macchina due volte l'anno.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import (ARCHIVE_CORE_FILES, ARCHIVE_MAX_AGE_DAYS, CLOSE_DIR,
                     DATA_DIR, DEFAULT_ARCHIVE_URL, FILE_METRICS, FILE_META,
                     FILE_UNIVERSE, SHARD_PATTERN, SHARD_SIZE)
from .metrics import COLONNE

VERSIONE_SCHEMA = 1

# Numero di fetta attribuito ai titoli che non hanno alcuna serie di prezzi.
# Restano nell'universo dello screener, ma non c'e' nessun file da aprire per
# loro: chi legge il pannello li salta.
SENZA_FETTA = -1


class ArchivioMancante(RuntimeError):
    """L'archivio non c'e' e non si e' potuto scaricare."""


# --------------------------------------------------------------------------
# Percorsi
# --------------------------------------------------------------------------
def cartella(base: Path | str | None = None) -> Path:
    return Path(base) if base is not None else DATA_DIR


def percorso_fetta(indice: int, base: Path | str | None = None) -> Path:
    return cartella(base) / CLOSE_DIR / SHARD_PATTERN.format(indice)


def _compressione() -> str:
    """zstd dove c'e', altrimenti snappy: la differenza e' un terzo di spazio."""
    try:
        import pyarrow.parquet  # noqa: F401
        from pyarrow import Codec

        if Codec.is_available("zstd"):
            return "zstd"
    except Exception:
        pass
    return "snappy"


# --------------------------------------------------------------------------
# Scrittura
# --------------------------------------------------------------------------
def assegna_fette(tickers: Sequence[str],
                  dimensione: int = SHARD_SIZE) -> dict[str, int]:
    """
    Decide in quale fetta finira' ogni titolo, PRIMA di scaricare qualsiasi cosa.

    L'assegnazione e' alfabetica e deterministica: due costruzioni dello
    stesso universo producono le stesse fette, il che rende gli archivi
    confrontabili e i download riprendibili.
    """
    ordinati = sorted(tickers)
    return {t: i // dimensione for i, t in enumerate(ordinati)}


def scrivi_fetta(serie: dict[str, tuple[np.ndarray, np.ndarray]],
                 indice: int, base: Path | str | None = None) -> int:
    """
    Salva una fetta del pannello prezzi.

    'serie' associa a ogni ticker la coppia (date, prezzi). Le date dei vari
    titoli non coincidono - chi e' quotato da vent'anni, chi da due - e
    vengono unite in un unico calendario, con NaN dove il titolo non era
    ancora quotato. E' esattamente quello che faceva pd.concat nel notebook,
    ma su un solo file invece che su millecinquecento.
    """
    percorso = percorso_fetta(indice, base)
    percorso.parent.mkdir(parents=True, exist_ok=True)

    if not serie:
        # Fetta vuota: si scrive comunque, cosi' il conteggio dei file torna.
        vuoto = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        vuoto.to_parquet(percorso, compression=_compressione())
        return 0

    calendario = np.unique(np.concatenate([d for d, _ in serie.values()]))
    indice_date = pd.DatetimeIndex(calendario, name="date")

    colonne = {}
    for ticker in sorted(serie):
        date, prezzi = serie[ticker]
        colonna = np.full(calendario.size, np.nan, dtype=np.float32)
        # Il calendario e' ordinato e contiene per costruzione tutte le date
        # di ogni titolo: searchsorted trova le posizioni in un colpo solo.
        # Con un dizionario servirebbero sette milioni di ricerche per fetta.
        colonna[np.searchsorted(calendario, date)] = prezzi
        colonne[ticker] = colonna

    df = pd.DataFrame(colonne, index=indice_date)
    df.to_parquet(percorso, compression=_compressione())
    return int(df.shape[1])


def riordina_per_anzianita(metriche: pd.DataFrame, mappa_vecchia: dict[str, int],
                           base: Path | str | None = None,
                           dimensione: int = SHARD_SIZE,
                           on_progress: Callable[[int, int], None] | None = None,
                           ) -> dict[str, int]:
    """
    Riscrive le fette mettendo davanti i titoli con la storia piu' lunga.

    Durante il download le fette si riempiono in ordine alfabetico, che e'
    l'unico ordine noto in partenza. Ma alfabetico e' l'ordine peggiore
    possibile per l'uso che se ne fara': i titoli che superano lo screener
    sono sparsi su tutte le lettere, quindi l'applicazione dovrebbe scaricare
    tutte le fette per simulare con un centinaio di titoli.

    Ordinandoli invece per anzianita' decrescente, i titoli che possono
    superare il filtro sullo storico - gli unici che la simulazione potra'
    mai usare - finiscono tutti nelle prime fette. Con vent'anni richiesti
    l'applicazione ne scarica due o tre invece di sedici.

    Il riordino costa qualche minuto di CI una volta ogni sei mesi, e fa
    risparmiare centocinquanta megabyte di scaricamento a ogni studente.
    """
    radice = cartella(base)
    if not mappa_vecchia or metriche.empty:
        return mappa_vecchia

    # Chi non ha una serie di prezzi non partecipa: e' rimasto nella tabella
    # delle misure (l'universo deve restare completo) ma non ha colonne.
    disponibili: dict[int, set[str]] = {}
    for indice in sorted(set(mappa_vecchia.values())):
        disponibili[indice] = set(colonne_fetta(indice, base))

    con_prezzi = {t for insieme in disponibili.values() for t in insieme}

    ordine = (metriche[["ticker", "anni_storico"]]
              .sort_values(["anni_storico", "ticker"],
                           ascending=[False, True], na_position="last")
              ["ticker"].astype(str).tolist())
    ordine = [t for t in ordine if t in con_prezzi]
    if not ordine:
        return mappa_vecchia

    mappa_nuova = {t: i // dimensione for i, t in enumerate(ordine)}
    n_nuove = max(mappa_nuova.values()) + 1

    temporanea = radice / (CLOSE_DIR + "_nuovo")
    if temporanea.exists():
        shutil.rmtree(temporanea)
    temporanea.mkdir(parents=True, exist_ok=True)

    for nuova in range(n_nuove):
        voluti = [t for t in ordine if mappa_nuova[t] == nuova]
        da_leggere: dict[int, list[str]] = {}
        for t in voluti:
            da_leggere.setdefault(mappa_vecchia[t], []).append(t)

        pezzi = []
        for vecchia in sorted(da_leggere):
            colonne = [t for t in da_leggere[vecchia]
                       if t in disponibili.get(vecchia, ())]
            if colonne:
                pezzi.append(pd.read_parquet(percorso_fetta(vecchia, base),
                                             columns=colonne))
        if not pezzi:
            continue

        unito = pd.concat(pezzi, axis=1).sort_index()
        unito = unito.astype(np.float32)
        unito.index.name = "date"
        unito.to_parquet(temporanea / SHARD_PATTERN.format(nuova),
                         compression=_compressione())
        del pezzi, unito

        if on_progress is not None:
            on_progress(nuova + 1, n_nuove)

    definitiva = radice / CLOSE_DIR
    shutil.rmtree(definitiva, ignore_errors=True)
    temporanea.rename(definitiva)
    return mappa_nuova


def scrivi_metriche(righe: Iterable[dict], mappa_fette: dict[str, int],
                    base: Path | str | None = None) -> pd.DataFrame:
    """
    Salva la tabella delle misure: una riga per ogni titolo dell'universo.

    E' il file su cui lavora lo screener. Ventitremila righe per una
    trentina di colonne stanno in pochi megabyte e si filtrano in
    millisecondi.
    """
    df = pd.DataFrame(list(righe))
    if df.empty:
        df = pd.DataFrame(columns=list(COLONNE))

    for colonna, tipo in COLONNE.items():
        if colonna not in df.columns:
            df[colonna] = np.nan
        if tipo.startswith("datetime"):
            df[colonna] = pd.to_datetime(df[colonna], errors="coerce")
        elif tipo == "string":
            df[colonna] = df[colonna].astype("string")
        else:
            df[colonna] = pd.to_numeric(df[colonna], errors="coerce").astype(tipo)

    df = df[list(COLONNE)]

    # Un titolo puo' esistere nell'universo senza avere alcuna serie di
    # prezzi: capita quando l'API risponde con una lista vuota. La sua riga
    # resta qui - l'universo dello screener dev'essere completo, altrimenti i
    # conteggi non tornano - ma non appartiene a nessuna fetta del pannello.
    # Il -1 e' esattamente quel "nessuna fetta", e i lettori lo sanno.
    df["fetta"] = df["ticker"].map(mappa_fette).fillna(SENZA_FETTA).astype("int16")
    df = df.sort_values("ticker").reset_index(drop=True)

    percorso = cartella(base) / FILE_METRICS
    percorso.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(percorso, compression=_compressione(), index=False)
    return df


def scrivi_universo(righe: Iterable[dict],
                    base: Path | str | None = None) -> pd.DataFrame:
    """Salva l'anagrafica dei simboli: codice, nome, tipo, borsa, valuta."""
    df = pd.DataFrame(list(righe))
    if df.empty:
        df = pd.DataFrame(columns=["ticker", "code", "name", "type",
                                   "exchange", "currency"])
    df = df.sort_values("ticker").reset_index(drop=True)
    percorso = cartella(base) / FILE_UNIVERSE
    percorso.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(percorso, compression=_compressione(), index=False)
    return df


def scrivi_meta(meta: dict, base: Path | str | None = None) -> dict:
    """Salva la carta d'identita' dell'archivio."""
    meta = dict(meta)
    meta.setdefault("versione_schema", VERSIONE_SCHEMA)
    meta.setdefault("costruito_il", dt.datetime.now(dt.timezone.utc).isoformat())
    percorso = cartella(base) / FILE_META
    percorso.parent.mkdir(parents=True, exist_ok=True)
    percorso.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return meta


# --------------------------------------------------------------------------
# Lettura
# --------------------------------------------------------------------------
def leggi_meta(base: Path | str | None = None) -> dict | None:
    percorso = cartella(base) / FILE_META
    if not percorso.exists():
        return None
    try:
        return json.loads(percorso.read_text(encoding="utf-8"))
    except Exception:
        return None


def leggi_metriche(base: Path | str | None = None) -> pd.DataFrame:
    percorso = cartella(base) / FILE_METRICS
    if not percorso.exists():
        raise ArchivioMancante(f"Manca il file delle misure: {percorso}")
    return pd.read_parquet(percorso)


def leggi_universo(base: Path | str | None = None) -> pd.DataFrame:
    percorso = cartella(base) / FILE_UNIVERSE
    if not percorso.exists():
        return pd.DataFrame(columns=["ticker", "name", "type", "exchange"])
    return pd.read_parquet(percorso)


def leggi_prezzi(tickers: Sequence[str], mappa_fette: dict[str, int],
                 base: Path | str | None = None) -> pd.DataFrame:
    """
    Legge le serie storiche dei soli titoli richiesti.

    Raggruppa i titoli per fetta, apre ciascuna fetta una volta sola e ne
    estrae le sole colonne che servono. Su un archivio di sedici fette, una
    selezione di duecento titoli ne tocca in genere tre o quattro.

    Restituisce un DataFrame date x ticker, allineato sul calendario unione
    delle fette lette, con NaN dove il titolo non era ancora quotato.
    """
    # Si scartano i titoli senza fetta: non hanno serie di prezzi, e cercarli
    # significherebbe aprire un file che non esiste.
    richiesti = [t for t in dict.fromkeys(tickers)
                 if int(mappa_fette.get(t, SENZA_FETTA)) >= 0]
    if not richiesti:
        return pd.DataFrame()

    per_fetta: dict[int, list[str]] = {}
    for t in richiesti:
        per_fetta.setdefault(int(mappa_fette[t]), []).append(t)

    pezzi = []
    for indice in sorted(per_fetta):
        percorso = percorso_fetta(indice, base)
        if not percorso.exists():
            continue
        disponibili = set(colonne_fetta(indice, base))
        volute = [t for t in per_fetta[indice] if t in disponibili]
        if not volute:
            continue
        pezzi.append(pd.read_parquet(percorso, columns=volute))

    if not pezzi:
        return pd.DataFrame()

    prezzi = pd.concat(pezzi, axis=1).sort_index()
    prezzi = prezzi.loc[:, [t for t in richiesti if t in prezzi.columns]]
    prezzi.index.name = "date"
    return prezzi


def colonne_fetta(indice: int, base: Path | str | None = None) -> list[str]:
    """Quali titoli contiene una fetta, senza leggerne i dati."""
    percorso = percorso_fetta(indice, base)
    if not percorso.exists():
        return []
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(percorso)
        return [n for n in schema.names if n != "date"]
    except Exception:
        return list(pd.read_parquet(percorso).columns)


def mappa_fette_da_metriche(metriche: pd.DataFrame) -> dict[str, int]:
    """Ricostruisce l'associazione ticker -> fetta dalla tabella delle misure."""
    if "fetta" not in metriche.columns:
        return {}
    return dict(zip(metriche["ticker"].astype(str), metriche["fetta"].astype(int)))


# --------------------------------------------------------------------------
# Stato dell'archivio
# --------------------------------------------------------------------------
def eta_giorni(meta: dict | None) -> float | None:
    """Da quanti giorni e' stato costruito l'archivio."""
    if not meta or not meta.get("costruito_il"):
        return None
    try:
        quando = dt.datetime.fromisoformat(str(meta["costruito_il"]))
    except ValueError:
        return None
    if quando.tzinfo is None:
        quando = quando.replace(tzinfo=dt.timezone.utc)
    delta = dt.datetime.now(dt.timezone.utc) - quando
    return delta.total_seconds() / 86400.0


def scaduto(meta: dict | None, soglia: int = ARCHIVE_MAX_AGE_DAYS) -> bool:
    eta = eta_giorni(meta)
    return eta is None or eta >= soglia


def archivio_presente(base: Path | str | None = None) -> bool:
    """Vero se ci sono almeno i file leggeri, quelli che bastano allo screener."""
    radice = cartella(base)
    return all((radice / nome).exists() for nome in ARCHIVE_CORE_FILES)


def stato(base: Path | str | None = None) -> dict:
    """Riepilogo dell'archivio locale, per la schermata di diagnostica."""
    radice = cartella(base)
    meta = leggi_meta(base)
    fette = sorted((radice / CLOSE_DIR).glob("part_*.parquet")) \
        if (radice / CLOSE_DIR).exists() else []
    peso = sum(p.stat().st_size for p in radice.rglob("*") if p.is_file())
    return {
        "cartella": str(radice),
        "presente": archivio_presente(base),
        "meta": meta,
        "eta_giorni": eta_giorni(meta),
        "scaduto": scaduto(meta),
        "fette_presenti": len(fette),
        "fette_attese": int(meta.get("n_fette", 0)) if meta else 0,
        "peso_mb": peso / (1024 * 1024),
    }


# --------------------------------------------------------------------------
# Recupero dell'archivio da rete
# --------------------------------------------------------------------------
def indirizzo_archivio() -> str:
    """
    Prefisso da cui scaricare l'archivio.

    Si legge dai secrets di Streamlit (ARCHIVE_URL) o dall'ambiente. Deve
    puntare alla release di GitHub prodotta dal workflow di aggiornamento.
    """
    import os

    url = None
    try:
        import streamlit as st

        url = st.secrets.get("ARCHIVE_URL")
    except Exception:
        url = None
    if not url:
        url = os.environ.get("KQ_ARCHIVE_URL")
    if not url:
        url = DEFAULT_ARCHIVE_URL
    return str(url).rstrip("/") if url else ""


def scarica_file(relativo: str, base: Path | str | None = None,
                 url_base: str | None = None,
                 on_progress: Callable[[int, int], None] | None = None) -> Path:
    """
    Scarica un pezzo dell'archivio e lo mette in cache su disco.

    Se il file c'e' gia' non fa nulla: e' il motivo per cui la seconda
    esecuzione dell'app parte all'istante.
    """
    import requests

    destinazione = cartella(base) / relativo
    if destinazione.exists() and destinazione.stat().st_size > 0:
        return destinazione

    prefisso = (url_base or indirizzo_archivio()).rstrip("/")
    if not prefisso:
        raise ArchivioMancante(
            "Nessun archivio in locale e nessun indirizzo da cui scaricarlo. "
            "Imposta ARCHIVE_URL nei secrets oppure costruisci l'archivio con "
            "scripts/build_dataset.py."
        )

    # Gli asset di una release di GitHub sono file piatti: la barra del
    # percorso interno diventa un trattino basso.
    nome_remoto = relativo.replace("/", "_").replace("\\", "_")
    url = f"{prefisso}/{nome_remoto}"

    destinazione.parent.mkdir(parents=True, exist_ok=True)
    temporaneo = destinazione.with_suffix(destinazione.suffix + ".parziale")

    with requests.get(url, stream=True, timeout=120) as risposta:
        if risposta.status_code != 200:
            raise ArchivioMancante(
                f"Non riesco a scaricare '{nome_remoto}' da {prefisso} "
                f"(codice {risposta.status_code})."
            )
        totale = int(risposta.headers.get("Content-Length", 0))
        fatti = 0
        with open(temporaneo, "wb") as f:
            for blocco in risposta.iter_content(chunk_size=1 << 20):
                if not blocco:
                    continue
                f.write(blocco)
                fatti += len(blocco)
                if on_progress is not None:
                    on_progress(fatti, totale)

    temporaneo.replace(destinazione)
    return destinazione


def assicura_nucleo(base: Path | str | None = None,
                    on_progress: Callable[[str, int, int], None] | None = None) -> None:
    """
    Garantisce la presenza dei file leggeri: meta, universo, misure.

    Bastano a far funzionare lo screener sull'intero universo. Le fette dei
    prezzi si scaricano solo quando servono davvero, cioe' al momento di
    simulare.
    """
    for nome in ARCHIVE_CORE_FILES:
        if (cartella(base) / nome).exists():
            continue
        scarica_file(
            nome, base,
            on_progress=(lambda f, t, n=nome: on_progress(n, f, t))
            if on_progress else None,
        )


def assicura_fette(indici: Iterable[int], base: Path | str | None = None,
                   on_progress: Callable[[str, int, int], None] | None = None) -> None:
    """Scarica le fette del pannello prezzi che ancora non sono in cache."""
    for indice in sorted({int(i) for i in indici if int(i) >= 0}):
        relativo = f"{CLOSE_DIR}/{SHARD_PATTERN.format(indice)}"
        if (cartella(base) / relativo).exists():
            continue
        scarica_file(
            relativo, base,
            on_progress=(lambda f, t, n=relativo: on_progress(n, f, t))
            if on_progress else None,
        )


def svuota_cache(base: Path | str | None = None) -> None:
    """Cancella l'archivio locale: il prossimo avvio lo riscarichera'."""
    radice = cartella(base)
    if radice.exists():
        shutil.rmtree(radice)
    radice.mkdir(parents=True, exist_ok=True)
