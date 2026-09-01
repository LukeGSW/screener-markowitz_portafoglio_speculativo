"""
Screener & Frontiera Efficiente - Scuola di Finanza Operativa.

Corso 2: Portafogli Avanzati.

I moduli sono divisi per responsabilita', e la divisione non e' decorativa:

    config      soglie, borse, palette. Non importa nulla di pesante, cosi'
                puo' essere letto anche dagli script da riga di comando.
    eodhd       tutto cio' che tocca la rete. Solo la costruzione
                dell'archivio lo usa: l'applicazione non chiama mai l'API.
    metrics     le misure di un titolo. E' il nucleo dello screener, ed e'
                l'unico posto in cui vivono le formule del notebook.
    screener    dalle misure alle soglie. Solo confronti fra colonne.
    datastore   scrittura, lettura e recupero dell'archivio Parquet.
    portfolio   Monte Carlo, ribilanciamenti, statistiche su finestra mobile.
    charts      grafici. Ricevono dati gia' calcolati e non calcolano nulla.
    texts       i testi didattici, perche' si possano cambiare senza
                toccare il codice.
    report      esportazione in HTML autonomo.
"""
