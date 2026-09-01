#!/usr/bin/env python
"""
Scrive le note della release a partire dal meta.json dell'archivio.

Vive in un file suo, e non dentro il workflow, per una ragione banale:
generare Markdown dentro una stringa YAML dentro uno script bash dentro una
sostituzione di comando e' il genere di cosa che funziona finche' non
funziona piu', e quando smette nessuno capisce perche'.

    python scripts/note_release.py dist/meta.json utente/repo > note.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def italiano(valore) -> str:
    """Migliaia separate dal punto, come si scrive dalle nostre parti."""
    try:
        return f"{int(valore):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(valore)


def main() -> int:
    if len(sys.argv) < 2:
        print("Uso: note_release.py <meta.json> [utente/repo]", file=sys.stderr)
        return 1

    meta = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    repository = sys.argv[2] if len(sys.argv) > 2 else "UTENTE/REPO"
    minuti = float(meta.get("download_secondi", 0)) / 60.0

    print(f"""Archivio dati dello **Screener & Frontiera Efficiente**.

L'applicazione lo scarica da sola: non c'e' nulla da installare a mano.

| | |
|---|---|
| Costruito il | {str(meta.get('costruito_il', ''))[:19].replace('T', ' ')} UTC |
| Universo | {meta.get('universo_etichetta', 'n.d.')} |
| Titoli esaminati | {italiano(meta.get('n_universo'))} |
| Titoli con serie storiche | {italiano(meta.get('n_con_prezzi'))} |
| Con {meta.get('anni_storico', '?')} anni di storia pieni | {italiano(meta.get('n_con_storico_pieno'))} |
| Finestra scaricata | {meta.get('data_inizio', '?')} - {meta.get('data_fine', '?')} |
| Fette del pannello prezzi | {meta.get('n_fette', '?')} |
| Durata del download | {minuti:.0f} minuti |

### Come si collega all'applicazione

Nei secrets di Streamlit Community Cloud (*Settings - Secrets*):

```toml
ARCHIVE_URL = "https://github.com/{repository}/releases/download/archivio-corrente"
```

Il tag e' mobile: a ogni ricostruzione questa release viene sostituita e
l'indirizzo resta lo stesso. Impostato una volta, non si tocca piu'.

### Che cosa contiene

`metrics.parquet` ha una riga per **ogni** titolo dell'universo, con tutte le
misure che i filtri confrontano gia' calcolate: e' il file che permette allo
screener di lavorare sull'intero listino in millesimi di secondo.
`close_part_NN.parquet` sono le serie storiche dei prezzi, divise in fette;
l'applicazione scarica solo quelle che le servono.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
