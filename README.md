# Screener & Frontiera Efficiente

**Scuola di Finanza Operativa — Percorso Investing**
Corso 2: *Portafogli Avanzati*

Conversione in applicazione web del notebook Colab del corso. Filtra l'intero
listino azionario americano con criteri di qualità e liquidità, estrae a sorte
decine di migliaia di portafogli equipesati fra i titoli sopravvissuti, e ne
misura rendimento, rischio e comportamento nei momenti brutti.

È il fratello del *Simulatore di Portafogli*: stessa impostazione, stessa
logica di dati, stessa grafica.

---

## Cosa fa

### Screening

- **Tutti** i titoli dell'universo, senza alcun pre-filtro — circa
  ventitremilacinquecento simboli fra azioni ordinarie ed ETF sul solo
  mercato americano, oltre ventottomila in modalità globale.
- **Otto filtri in cascata**: anzianità dello storico, prezzo minimo e
  massimo, volume medio in dollari, performance continua su finestra
  pluriennale, drawdown massimo, Sharpe ratio, volatilità minima.
- **Risposta immediata a ogni movimento di cursore.** Non c'è alcun pulsante
  "applica": si sposta una soglia e l'universo si ricalcola sotto gli occhi.
- **Tabella dei motivi di scarto**: quanti titoli sono caduti su ciascun
  filtro, nell'ordine di applicazione. È il modo più rapido per capire quale
  vincolo stia davvero decidendo la selezione.
- **Curva di sensibilità**: quanti titoli sopravvivono al variare di una sola
  soglia, tenendo ferme tutte le altre.
- **L'universo intero nel piano rischio-rendimento**, con i promossi in
  evidenza: si vede a colpo d'occhio quale angolo della nuvola i filtri stiano
  ritagliando.

### Simulazione

- Monte Carlo da 5.000 a 500.000 portafogli equipesati, con numero di titoli
  variabile fra un minimo e un massimo.
- **Frontiera efficiente interattiva** con i tre portafogli notevoli: massimo
  Sharpe, minima volatilità, massimo rendimento a diversificazione vincolata.
- Seme casuale impostabile: stesso seme, stessi portafogli. In aula serve.
- Controllo di profondità storica prima della simulazione — la "caccia
  all'intruso" del notebook — con l'elenco dei titoli esclusi e il perché.

### Report

- Curve del capitale dei tre portafogli contro l'SPY, in scala lineare o
  logaritmica.
- Grafico *sott'acqua*: quanto si è stati sotto il massimo precedente, giorno
  per giorno.
- **Statistiche su ogni possibile periodo pluriennale** — 1, 2, 3 e 5 anni —
  con rendimento e drawdown minimo, medio, massimo e mediano.
- Matrice di correlazione dei titoli più ricorrenti fra i portafogli migliori.
- I singoli titoli di un portafoglio, uno per uno, per vedere la dispersione
  che si nasconde dentro una media.
- **Esportazione in HTML autonomo**, con i grafici veri e non immagini: si
  apre in qualunque browser, senza Python, e si stampa in PDF.

### Cinque metodi di ribilanciamento

Pesi costanti (che equivale a ribilanciare ogni sera), buy & hold vero,
trimestrale, semestrale e annuale — sull'ultimo giorno **di borsa** del
periodo, non sulla data di calendario.

---

## Perché è veloce

Il notebook era lento per una ragione strutturale, non per disattenzione:
**rileggeva e rimisurava tutto a ogni cambio di parametro.** Ventitremila
file CSV, dodici gigabyte, ogni volta che si spostava una soglia.

Ma le misure di un titolo — da quanto è quotato, quanto scambia, quanto ha
perso nel peggior momento, che Sharpe ha avuto — **non dipendono dalle
soglie.** Dipendono solo dai prezzi. Quindi si calcolano una volta sola,
quando si costruisce l'archivio, e si mettono in tabella.

Da quel momento filtrare l'universo intero è un confronto fra colonne di
NumPy.

| | Notebook | Applicazione | |
|---|---|---|---|
| Rilettura dei dati a ogni cambio di soglia | ~8 min (12,4 GB di CSV) | nessuna | — |
| Screening di 23.479 titoli | ~5 s di sola CPU, oltre alla rilettura | **39 ms** | **125×** |
| Monte Carlo, 50.000 portafogli | 2,3 s | **0,13 s** | **18×** |
| Statistiche rolling, 4 strategie | 26 s | **0,32 s** | **81×** |
| Costruzione dell'archivio | — | ~35 min, due volte l'anno, su GitHub | — |

*Misure su questa macchina; il rapporto conta più dei valori assoluti. Il
tempo di rilettura dei CSV è misurato su SSD locale: su Google Drive montato
in Colab era considerevolmente peggiore.*

Le tre riscritture che producono questi numeri:

1. **Le misure si calcolano una volta.** Compresi i rendimenti su finestra
   mobile per tutte le durate da uno a dieci anni, così anche il cursore
   della performance continua risponde all'istante.
2. **Il Monte Carlo è vettoriale.** Le cinquantamila estrazioni sono una
   manciata di operazioni NumPy a blocchi, invece di un ciclo Python con
   cinquantamila piccoli prodotti matriciali. La matematica è identica.
3. **I drawdown su finestra mobile usano una vista scorrevole.** Il notebook
   chiamava `.rolling().apply()` con una funzione Python una volta per ogni
   giorno di ogni finestra: era di gran lunga la parte più lenta della
   visualizzazione.

E il download, che resta lento perché è rete e non calcolo, è uscito
dall'applicazione: lo fa una macchina di GitHub due volte l'anno.

---

## L'archivio

L'applicazione **non scarica dati mentre la si usa**. Legge un archivio
costruito in precedenza, che contiene:

| File | Contenuto | Peso |
|---|---|---|
| `meta.json` | quando e come è stato costruito | 1 KB |
| `universe.parquet` | anagrafica dei simboli | ~1 MB |
| `metrics.parquet` | **una riga per ogni titolo**, con tutte le misure già calcolate | ~6 MB |
| `close_part_NN.parquet` | serie storiche dei prezzi, in fette da 1.500 titoli | ~150-250 MB in tutto |

I primi tre si scaricano all'avvio e bastano a far funzionare lo screener
sull'universo intero. Le fette dei prezzi arrivano **solo quando si simula**,
e solo quelle che contengono i titoli sopravvissuti ai filtri: in genere due
o tre.

### Perché si ricostruisce da zero, e non un pezzo per volta

Perché l'*adjusted close* — il prezzo corretto per dividendi e frazionamenti,
l'unico su cui abbia senso misurare un rendimento pluriennale — viene
**ricalcolato all'indietro su tutta la serie** ogni volta che c'è un nuovo
stacco o un nuovo frazionamento. Accodare gli ultimi sei mesi a una serie
scaricata sei mesi fa produrrebbe un gradino artificiale nel punto di
giunzione, e quel gradino finirebbe dritto nel calcolo del drawdown massimo.

Mezz'ora di macchina due volte l'anno è il prezzo della correttezza.

---

## Aggiornamento automatico ogni sei mesi

Il workflow `aggiorna-archivio` ricostruisce l'archivio e lo pubblica come
release di GitHub. Serve una sola cosa per attivarlo:

> **Settings → Secrets and variables → Actions → New repository secret**
> Nome: `EODHD_API_KEY` — Valore: la tua chiave

### Come è organizzato, e perché

Il cron **scatta ogni mese**, non ogni sei. Non è una svista: GitHub
disattiva d'ufficio le pianificazioni dei repository fermi da sessanta
giorni, e fra due esecuzioni semestrali ne passano centottanta — la seconda
non partirebbe mai.

Le esecuzioni mensili non fanno quasi nulla. Lo script confronta la data
dell'ultima costruzione con la soglia di 180 giorni e, se l'archivio è ancora
giovane, esce in pochi secondi **senza consumare una sola chiamata API**.
L'undicesima volta ricostruisce davvero. E poiché ogni costruzione riscrive
`archivio/meta.json` nel repository, l'attività non si azzera mai e la
pianificazione resta viva.

### Per ricostruirlo subito

**Actions → aggiorna-archivio → Run workflow.** Il campo *forza* è già
attivo: l'avvio manuale ricostruisce sempre, anche se l'archivio è di ieri.
Utile prima di una lezione, o quando si vuole cambiare universo.

### Quanto costa

Una chiamata API per titolo: circa **23.500 chiamate** per il mercato
americano, in una sessione sola, due volte l'anno. Il piano *All-In-One* di
EODHD ne concede centomila al giorno. A 900 chiamate al minuto il download
dura poco più di mezz'ora; il tetto di GitHub per un singolo job è di sei ore.

Chi ha un piano più generoso può alzare il ritmo con la variabile d'ambiente
`KQ_RATE_PER_MIN`.

---

## Come metterlo online in dieci minuti

### 1. Chiave API EODHD

Serve un account su [eodhd.com](https://eodhd.com) con copertura del mercato
statunitense (`.US`). La chiave è **una sola, quella del docente**: gli
studenti usano l'applicazione pubblicata e non devono aprire alcun account.
Non esiste alcun campo di inserimento nell'interfaccia.

### 2. Carica il progetto su GitHub

```bash
git init && git add . && git commit -m "Screener & Frontiera Efficiente" && git branch -M main
```

Poi collega il repository remoto e fai `git push`.

> **Importante.** Non caricare mai `.streamlit/secrets.toml` con la chiave
> vera: è già escluso dal `.gitignore`. Nel repository deve finire solo
> `secrets.toml.example`.

### 3. Costruisci il primo archivio

Aggiungi il segreto `EODHD_API_KEY` al repository, poi **Actions →
aggiorna-archivio → Run workflow**. Dopo circa mezz'ora trovi la release
`archivio-corrente` con dentro tutti i file.

### 4. Pubblica su Streamlit Community Cloud

1. [share.streamlit.io](https://share.streamlit.io), accedi con GitHub.
2. *Create app* → repository, branch `main`, file `app.py`.
3. Prima di *Deploy*, apri **Advanced settings → Secrets** e incolla
   l'indirizzo che trovi nelle note della release:

   ```toml
   ARCHIVE_URL = "https://github.com/TUO_UTENTE/TUO_REPO/releases/download/archivio-corrente"
   ```

4. *Deploy*.

Il tag `archivio-corrente` è **mobile**: a ogni ricostruzione la release viene
sostituita e l'indirizzo resta identico. Impostato una volta, non si tocca
più.

---

## Uso in locale

```bash
pip install -r requirements.txt
```

```bash
export EODHD_API_KEY="la_tua_chiave"
python scripts/build_dataset.py --universo USA --anni 20 --out data
streamlit run app.py
```

Sono circa venticinque minuti e ventiquattromila chiamate API. Per una prova
di funzionamento, `--limite 400` scarica solo i primi quattrocento titoli:
resta un archivio di quotazioni **vere**, semplicemente parziale.

L'applicazione si apre su `http://localhost:8501`.

---

## L'archivio o è vero, o non si apre

Non esiste alcun comando, in questo progetto, che generi un archivio di dati
finti. È una scelta, e la ragione è semplice: **un archivio sintetico non si
distingue da uno vero guardando i grafici.** Le curve sono plausibili, le
statistiche credibili, i portafogli hanno l'aria sensata. Se ne accorgerebbe
solo chi controlla i ticker uno per uno, e nessuno lo fa — men che meno
durante una lezione.

Quindi il controllo è a monte e non è aggirabile per distrazione:

- `build_dataset.py` scrive nel `meta.json` il campo `"origine": "eodhd"`, e
  quella riga viene eseguita **solo dopo un download vero**;
- l'applicazione, prima di caricare qualunque dato, verifica quel campo. Se
  manca o dice altro, **si ferma** con una schermata che spiega perché e come
  rimediare. Non mostra un avviso: rifiuta di partire;
- gli unici dati sintetici del progetto vivono dentro `tests/fixture.py`,
  nascono in una cartella temporanea del sistema, dichiarano
  `"origine": "test"` e vengono distrutti a fine test. Anche se qualcuno li
  copiasse in `data/`, l'applicazione li respingerebbe.

Il test d'insieme verifica **anche il lucchetto**: fallisce se l'archivio di
prova venisse per qualunque motivo accettato come autentico.

---

## Controlli automatici

```bash
python tests/test_equivalenza.py     # gli stessi verdetti del notebook
python tests/test_pipeline.py        # dall'archivio al report, senza rete
```

Il primo è quello che conta. Riscrivere un calcolo per renderlo cento volte
più rapido serve a poco se poi seleziona titoli diversi: il test copia
**verbatim** le funzioni della Cella 3 del notebook, le esegue su
quattrocento titoli sintetici con tre configurazioni di soglie diverse, e
verifica che ogni singolo titolo riceva lo stesso identico verdetto. Fa lo
stesso con il Monte Carlo (confrontato con il calcolo portafoglio per
portafoglio) e con le statistiche rolling (confrontate con
`.rolling().apply()`).

Girano anche a ogni push, nel workflow `controlli`.

---

## Struttura del progetto

```
screener-markowitz/
├── app.py                     interfaccia Streamlit (cinque schede)
├── requirements.txt
├── .gitignore                 esclude secrets.toml e l'archivio dati
├── .streamlit/
│   ├── config.toml            tema scuro
│   └── secrets.toml.example   modello per chiave e indirizzo archivio
├── src/
│   ├── config.py              soglie, borse, palette, percorsi
│   ├── eodhd.py               client API, limitatore, download parallelo
│   ├── metrics.py             misure di un titolo — il nucleo dello screener
│   ├── screener.py            dalle misure alle soglie
│   ├── datastore.py           scrittura, lettura e recupero dell'archivio
│   ├── portfolio.py           Monte Carlo, ribilanciamenti, rolling
│   ├── charts.py              grafici Plotly
│   ├── texts.py               testi didattici
│   └── report.py              esportazione HTML
├── scripts/
│   ├── build_dataset.py       costruzione dell'archivio (è ciò che gira in CI)
│   └── note_release.py        note della release da meta.json
├── tests/
│   ├── fixture.py             materiale sintetico, solo per i test
│   ├── test_equivalenza.py    stessi verdetti del notebook
│   └── test_pipeline.py       prova d'insieme, dall'archivio al report
└── .github/workflows/
    ├── aggiorna-archivio.yml  ricostruzione semestrale
    └── controlli.yml          test a ogni push
```

Per cambiare un testo dell'interfaccia si interviene su `src/texts.py`; per
soglie di partenza, borse o colori, su `src/config.py`.

---

## Che cosa cambia rispetto al notebook

I verdetti dello screener e la matematica dei portafogli sono identici — c'è
un test che lo dimostra. Cambiano cinque cose, tutte deliberate.

**I titoli non misurabili vengono scartati, non promossi.** Nel notebook i
filtri erano scritti come `if valore < soglia: scarta`, e in NumPy
`NaN < soglia` è **falso**: un titolo con dati corrotti, di cui non si
riusciva a calcolare drawdown o Sharpe, passava indenne tutti i controlli.
Ora viene scartato con motivazione *dati insufficienti*.

**I ribilanciamenti avvengono davvero.** Il notebook confrontava date di
calendario (`31 dicembre`) con l'indice dei prezzi: il 31 dicembre 2022 era
un sabato, e quel ribilanciamento semplicemente non avveniva. Ora si
ribilancia l'ultimo giorno **di borsa** del periodo.

**Le serie dei portafogli hanno tutte la stessa lunghezza.** Nel notebook il
ramo senza ribilanciamento restituiva un giorno in più degli altri, e le
curve del capitale partivano da punti diversi.

**"Nessun ribilanciamento" ha cambiato nome.** L'opzione `NONE` del notebook
teneva i pesi costanti *ogni giorno*, che è l'esatto contrario di non fare
nulla: equivale a ribilanciare tutte le sere. Il comportamento è stato
mantenuto identico — è quello con cui sono stati prodotti i risultati del
corso — ma ora si chiama *pesi costanti*, e accanto è stato aggiunto un
**buy & hold vero**, che compra quote uguali il primo giorno e non tocca più
nulla. Il confronto fra i due è l'esercizio più istruttivo sul
ribilanciamento.

**I prezzi non positivi o non finiti vengono eliminati alla fonte**, quando la
serie arriva da EODHD, invece di propagarsi nei calcoli.

---

## Che cosa non è modellato

**Costi.** Niente commissioni, spread o impatto di mercato. Su un portafoglio
ribilanciato ogni giorno — l'impostazione predefinita — è un'assunzione molto
forte, e va tenuta a mente prima di confrontare il ribilanciamento
giornaliero con il buy & hold.

**Fiscalità.** Nessuna imposta su plusvalenze o dividendi. Il ribilanciamento
a vendita è un evento fiscale ogni volta che l'asset venduto è in guadagno.

**Titoli usciti dal listino.** L'archivio contiene i simboli quotati **oggi**.
Le società fallite, fuse o ritirate non ci sono. È la forma più classica di
*survivorship bias*, e da sola basta a spiegare una parte non piccola dei
rendimenti che si vedranno.

**Il fatto che la selezione guardi al passato.** I titoli sono scelti perché
si sono comportati bene in un periodo, e poi misurati **su quello stesso
periodo**. È il difetto più grave dell'esercizio ed è insanabile per
costruzione: l'unico modo serio di affrontarlo sarebbe selezionare sui primi
dieci anni e misurare sui secondi dieci. È anche la ragione per cui
l'esercizio si chiama *portafogli speculativi* e non *piano di investimento*.

---

## Avvertenza

Questo è uno **strumento didattico**. I risultati sono simulazioni su dati
storici: i rendimenti passati non sono in alcun modo indicativi di quelli
futuri. I portafogli prodotti sono **esempi di montaggio, non
raccomandazioni**. Nessun contenuto dell'applicazione costituisce consulenza
finanziaria o raccomandazione personalizzata.

Prima di investire, verifica i costi effettivi applicati dal tuo
intermediario e il trattamento fiscale applicabile nel tuo paese, e valuta se
gli strumenti siano adatti alla tua situazione, eventualmente con
l'assistenza di un consulente abilitato.
