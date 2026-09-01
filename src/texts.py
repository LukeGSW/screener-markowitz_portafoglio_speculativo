"""
Testi didattici dell'applicazione.

Sono i commenti del notebook, riportati qui e ampliati. Vivono in un file a
parte per una ragione pratica: un docente che vuole cambiare una spiegazione
lo fa qui, senza mai toccare il codice di calcolo.

Sono scritti in Markdown. Streamlit li rende direttamente; il report
esportabile li converte in HTML con il piccolo traduttore in report.py.
"""

from __future__ import annotations

# ==========================================================================
# Cornice
# ==========================================================================
DISCLAIMER = """
Questo e' uno **strumento didattico**. I risultati sono simulazioni su dati
storici, e i rendimenti passati non sono in alcun modo indicativi di quelli
futuri. I portafogli che l'applicazione produce sono **esempi di montaggio,
non raccomandazioni**: nessun contenuto costituisce consulenza finanziaria o
raccomandazione personalizzata.

C'e' poi un avvertimento specifico di questo esercizio, e va preso sul serio.
Selezionare titoli sulla base di come si sono comportati negli ultimi
vent'anni, e poi misurare quei titoli **sugli stessi vent'anni**, produce
risultati bellissimi e per costruzione ottimistici. Si chiama *survivorship
bias* quando riguarda i titoli sopravvissuti, *look-ahead bias* quando
riguarda l'uso di informazioni che all'epoca non erano disponibili: qui ci
sono entrambi, e sono la ragione per cui l'esercizio si chiama "portafogli
speculativi" e non "piano di investimento".
"""

INTRO = """
Questa applicazione fa in tre schermate quello che il notebook del corso
faceva in sei celle: **filtra** l'intero listino americano con criteri di
qualita' e liquidita', **estrae a sorte** decine di migliaia di portafogli
equipesati fra i titoli sopravvissuti, e ne **misura** rendimento, rischio e
comportamento nei momenti brutti.

Il metodo e' quello di Markowitz, con una torsione. Markowitz cerca i pesi
ottimali di un insieme di titoli dato; qui i pesi sono fissi e uguali, e la
ricerca riguarda **quali titoli mettere dentro**. E' un'inversione
deliberata: sui dati storici la scelta dei componenti pesa piu' della
scelta dei pesi, e i pesi ottimali sono notoriamente instabili.
"""

COME_SI_USA = """
**1. Dati** - verifica che l'archivio ci sia e da quando e' fermo.
**2. Screener** - muovi le soglie e guarda l'universo restringersi in tempo
reale. La tabella dei motivi di scarto dice quale filtro sta decidendo.
**3. Simulazione** - lancia il Monte Carlo sui titoli promossi e ottieni la
frontiera efficiente con i tre portafogli notevoli.
**4. Report** - le curve del capitale, i drawdown, le statistiche su ogni
possibile periodo pluriennale, le correlazioni. Esportabile in HTML.
"""

# ==========================================================================
# L'archivio
# ==========================================================================
ARCHIVIO = """
### Perche' i dati sono fermi a una data

L'universo azionario americano conta circa **ventitremilacinquecento**
simboli fra azioni ordinarie ed ETF. Scaricare vent'anni di quotazioni per
ognuno significa altrettante chiamate all'API: mezz'ora di lavoro di
macchina e piu' di dieci gigabyte di dati grezzi.

Farlo davanti allo studente non ha alcun senso. Si fa **una volta ogni sei
mesi**, di notte, su una macchina di GitHub, e il risultato e' un archivio
che l'applicazione legge gia' pronto. Da quel momento lo screening
dell'intero universo e' questione di millesimi di secondo, perche' tutte le
misure che i filtri confrontano - anzianita', liquidita', drawdown, Sharpe,
volatilita', rendimenti pluriennali - **sono gia' state calcolate** e stanno
in una tabella di ventitremila righe.

### Perche' non si aggiorna un pezzo per volta

Perche' non funzionerebbe. Il prezzo su cui si misura tutto e'
l'*adjusted close*, cioe' il prezzo corretto per dividendi e frazionamenti,
e quella correzione viene **riapplicata all'indietro su tutta la serie** ogni
volta che c'e' un nuovo stacco o un nuovo frazionamento. Accodare gli ultimi
sei mesi a una serie scaricata sei mesi fa produrrebbe un gradino artificiale
nel punto di giunzione, e quel gradino finirebbe dritto nel calcolo del
drawdown massimo. Si ricostruisce da zero, ed e' l'unica cosa corretta da
fare.

### Che cosa cambia in sei mesi

Poco, e in modo prevedibile. Un titolo in piu' o in meno che supera i venti
anni di storia, qualche variazione nei volumi medi, i rendimenti dell'ultimo
semestre che entrano nei calcoli. Se durante il corso serve una fotografia
piu' fresca, l'aggiornamento si puo' lanciare a mano in qualunque momento.
"""

# ==========================================================================
# Lo screener
# ==========================================================================
SCREENER = """
### Che cosa fanno gli otto filtri

I filtri si applicano **in cascata**, nell'ordine in cui sono elencati: ogni
titolo viene attribuito al primo criterio che lo esclude. E' il motivo per
cui la tabella dei motivi di scarto e' leggibile: dice quanti titoli sono
caduti su ciascun filtro, non quanti *sarebbero* caduti se fossero arrivati
fin li'.

**Anzianita' dello storico.** Il titolo deve essere quotato da almeno il
numero di anni richiesto. E' quasi sempre il filtro che decide tutto: la
grande maggioranza dei simboli americani e' piu' giovane di vent'anni, fra
quotazioni recenti, ETF nati dopo il 2010 e societa' arrivate in borsa da
poco. Non e' un capriccio: serve che tutti i titoli confrontati abbiano
attraversato gli stessi eventi, il 2008 compreso.

**Prezzo minimo e massimo.** Sotto una certa soglia - cinque dollari per
convenzione - si entra nel territorio delle *penny stock*, dove lo spread
denaro-lettera vale percentuali intere e il prezzo si muove per ragioni che
non hanno a che vedere con l'azienda. Il tetto massimo serve solo a
escludere le anomalie di quotazione.

**Volume in dollari.** Prezzo per pezzi scambiati, mediato sugli ultimi
novanta giorni di borsa. E' il filtro piu' sottovalutato e uno dei piu'
importanti: un titolo che scambia centomila dollari al giorno puo' avere
statistiche magnifiche ed essere del tutto inutilizzabile, perche' nel
momento in cui si prova a venderne una quantita' seria il prezzo si muove
contro. Il valore di partenza, dieci milioni al giorno, e' prudente.

**Performance continua.** Si guarda ogni giorno della storia e lo si
confronta con il giorno di N anni prima. Se **anche il peggiore** di questi
confronti e' positivo, vuol dire che nella storia del titolo non e' mai
esistito un periodo di quella lunghezza chiuso in perdita. E' un criterio
molto severo - e molto selettivo - che premia la regolarita' piu' del
rendimento.

**Drawdown massimo.** La perdita piu' profonda dal massimo precedente. Un
titolo che nel 2008 e' sceso del novanta per cento e poi ha recuperato ha
un rendimento complessivo ottimo e un profilo che quasi nessuno riesce a
sopportare davvero.

**Sharpe ratio.** Rendimento in eccesso rispetto al tasso privo di rischio,
diviso per la volatilita'. E' la misura di quanto si e' stati pagati per il
rischio corso.

**Volatilita' minima.** Il solo filtro che chiede *piu'* movimento, non
meno, e ha senso solo in questo esercizio: un portafoglio dichiaratamente
speculativo non ha nulla da fare con titoli che non si muovono.
"""

IMBUTO = """
Questa tabella e' il posto in cui guardare quando l'universo risulta vuoto o
troppo affollato. La barra piu' lunga e' il vincolo che sta effettivamente
decidendo la selezione: allentare qualunque altro filtro non cambiera'
quasi nulla.

Nove volte su dieci la barra piu' lunga e' l'anzianita' dello storico, ed e'
una cosa che vale la pena vedere con i propri occhi: la sensazione diffusa
e' che a selezionare siano i criteri di qualita', mentre in realta' a
selezionare e' quasi sempre l'anagrafe.
"""

NUVOLA = """
Ogni punto e' un titolo dell'universo: in orizzontale quanto oscilla, in
verticale quanto e' cresciuto ogni anno. In grigio gli scartati, in colore
i promossi.

Serve a rendersi conto di che cosa i filtri stiano effettivamente
scegliendo. Quasi sempre e' un angolo molto piccolo e molto particolare
della nuvola - alto e a sinistra - e guardarlo e' il modo piu' rapido per
accorgersi se le soglie stanno facendo quello che si crede, oppure se si
sta selezionando per caso qualcosa di diverso.
"""

# ==========================================================================
# La simulazione
# ==========================================================================
FRONTIERA = """
### Come si legge

Il grafico della frontiera efficiente e' la rappresentazione centrale della
teoria di Markowitz. Ogni punto e' un portafoglio: sull'asse **orizzontale**
la volatilita' annualizzata, cioe' il rischio; sull'asse **verticale** il
rendimento annualizzato; il **colore** dice lo Sharpe ratio, cioe' il
rapporto fra le due cose.

La **frontiera efficiente** e' il bordo superiore sinistro della nuvola. I
portafogli che stanno li' sono efficienti nel senso preciso che non esiste
un'alternativa che renda di piu' a parita' di rischio, ne' una che rischi
meno a parita' di rendimento. Tutto quello che sta sotto e a destra e'
dominato: si puo' fare meglio senza rinunciare a nulla.

I tre portafogli notevoli sono marcati con simboli grandi.

### Che cosa questo grafico non dice

Due cose, ed entrambe contano.

La prima: la nuvola e' fatta di portafogli **equipesati**, estratti a sorte
fra i titoli promossi. Non e' l'insieme di tutti i portafogli possibili, ed
e' molto meno ampia della frontiera che si otterrebbe ottimizzando anche i
pesi. E' una scelta voluta - i pesi ottimizzati sono instabili e si
appoggiano a stime che cambiano ogni volta che si sposta la finestra di
osservazione - ma va saputa.

La seconda: e' tutto **calcolato all'indietro**. La frontiera di domani non
somigliera' a questa. Il rendimento atteso e' la media storica, e la media
storica e' una stima pessima del futuro; la matrice di covarianza e' un po'
piu' stabile, ma non abbastanza da fidarsi ciecamente.
"""

PORTAFOGLI_CHIAVE = """
Dall'insieme dei portafogli simulati se ne isolano tre, che rispondono a tre
domande diverse.

**Massimo Sharpe ratio.** Il portafoglio meglio pagato per unita' di
rischio. E' quello che la teoria indica come "ottimale" per chi puo'
regolare la propria esposizione aggiungendo o togliendo liquidita'.

**Minima volatilita'.** Il portafoglio che ha oscillato di meno, senza alcun
riguardo per quanto abbia reso. Interessa a chi mette la conservazione del
capitale davanti a tutto.

**Massimo rendimento vincolato.** Il portafoglio che ha reso di piu' fra
quelli con almeno un numero minimo di titoli. Il vincolo non e' un
dettaglio tecnico: senza di esso vincerebbe sempre il portafoglio piu'
concentrato, che e' un modo elaborato per dire "il titolo che e' salito di
piu'". Con il vincolo si guarda almeno un portafoglio, non una scommessa.

Confrontarli fra loro e con l'SPY e' l'esercizio piu' istruttivo
dell'applicazione: quasi sempre il piu' redditizio e' anche quello con il
drawdown peggiore, e la differenza fra i tre e' meno grande di quanto ci si
aspetti.
"""

TABELLA_TOP = """
I portafogli con lo Sharpe ratio piu' alto, in classifica. Per ciascuno la
composizione esatta, il numero di titoli, rendimento, volatilita' e Sharpe.

Vale la pena leggere la colonna dei titoli piu' che quella dello Sharpe: si
scopre in genere che i primi quindici portafogli sono **quasi lo stesso
portafoglio**, con due o tre titoli che ruotano. E' un'informazione utile,
perche' significa che la differenza di Sharpe fra il primo e il quindicesimo
non e' una differenza di strategia ma di rumore.
"""

EQUITY = """
La curva del capitale mostra quanto sarebbe diventato un euro investito
all'inizio, giorno per giorno, per ciascun portafoglio e per il benchmark.

Su vent'anni conviene guardarla in **scala logaritmica**: in scala lineare
il primo decennio si schiaccia sull'asse e sembra che non sia successo
nulla, mentre in scala logaritmica la stessa pendenza significa lo stesso
rendimento percentuale in qualunque punto del grafico.

Le cose da guardare, nell'ordine: dove si arriva rispetto all'SPY; **come**
ci si arriva, cioe' se la curva sale regolare o a strappi; e che cosa e'
successo nel 2008, nel 2020 e nel 2022, che sono i tre momenti in cui si
decide se una strategia e' sopportabile.
"""

DRAWDOWN = """
Il grafico "sott'acqua" mostra, giorno per giorno, quanto si sta sotto il
massimo precedente. Vale zero quando il portafoglio e' su un nuovo massimo,
e scende in territorio negativo per tutto il tempo del recupero.

E' il grafico che andrebbe guardato **per primo**, prima ancora della curva
del capitale. La curva del capitale dice dove si e' arrivati; questo dice
che cosa si e' dovuto sopportare per arrivarci, ed e' l'unica delle due
cose che si vive davvero mentre accade.

Tre cose da leggere: la **profondita'** del punto piu' basso, la **durata**
dei periodi sott'acqua - un meno venti per cento che dura quattro anni e'
molto piu' difficile di un meno trenta per cento riassorbito in sei mesi - e
la **frequenza** con cui la curva torna a zero.
"""

ROLLING = """
### Perche' non basta il rendimento complessivo

Il rendimento di vent'anni dipende in modo pesante da **quando** si e'
entrati. Un portafoglio che ha fatto il nove per cento l'anno dal 2005 al
2025 puo' avere fatto il venti per cento l'anno per chi e' entrato nel 2009
e il due per cento per chi e' entrato nel 2007.

Queste tabelle guardano **tutti i possibili punti di ingresso**. Per ogni
finestra - un anno, due, tre, cinque - si calcola il rendimento totale e il
drawdown massimo di ciascun periodo di quella lunghezza contenuto nella
storia, e se ne riportano media, minimo, massimo e mediana.

### Come si leggono

**Rendimento minimo** e' il numero piu' importante della tabella: e' quanto
avrebbe perso chi fosse entrato nel momento peggiore possibile e uscito
esattamente N anni dopo. Se il rendimento minimo a cinque anni e' positivo,
significa che in tutta la storia esaminata non e' mai esistito un
quinquennio chiuso in perdita.

**DD massimo** e' il drawdown piu' severo osservato dentro una finestra di
quella lunghezza: e' il peggio che si sarebbe dovuto sopportare restando
investiti per quel periodo.

La distanza fra media e mediana dice quanto la distribuzione sia sbilanciata.
Quando la media e' molto piu' alta della mediana, il rendimento medio e'
prodotto da pochi periodi eccezionali e non e' quello che ci si deve
aspettare da un ingresso qualunque.
"""

CORRELAZIONI = """
La matrice mostra quanto si muovono insieme i titoli che compaiono piu'
spesso nei portafogli con lo Sharpe migliore.

Ogni cella e' il coefficiente di correlazione fra due titoli: **+1** vuol
dire che si muovono all'unisono, **0** che non hanno relazione lineare,
**-1** che vanno in direzioni opposte. La diagonale vale sempre 1.

Ai fini della diversificazione si cercano valori **bassi o negativi**: se
gli asset non si muovono insieme, la perdita su uno puo' essere compensata
dal guadagno su un altro e la volatilita' complessiva scende sotto la media
di quelle individuali. Se invece la matrice e' tutta rossa, la
diversificazione e' apparente: si hanno cinque titoli e una sola scommessa,
ripetuta cinque volte.

E' un esito frequente in questo esercizio, e non e' un caso: filtri che
premiano crescita regolare e drawdown contenuto tendono a selezionare
titoli che assomigliano fra loro.
"""

# ==========================================================================
# Aiuti puntuali
# ==========================================================================
FILTRI_HELP = {
    "anni_storico": "Da quanti anni il titolo deve essere quotato. E' quasi "
                    "sempre il filtro che decide la selezione.",
    "anni_perf_continua": "Il titolo non deve avere alcun periodo di questa "
                          "lunghezza chiuso in perdita, in tutta la sua storia.",
    "max_drawdown": "La perdita massima dal massimo precedente che si e' "
                    "disposti a tollerare sul singolo titolo.",
    "min_sharpe": "Rendimento in eccesso per unita' di rischio. Sopra 1 e' "
                  "raro su vent'anni, sopra 0,5 gia' selettivo.",
    "min_volatilita": "Volatilita' MINIMA, calcolata sui rendimenti mensili. "
                      "Serve a escludere i titoli che non si muovono.",
    "prezzo_minimo": "Sotto i cinque dollari si entra nel territorio delle "
                     "penny stock, dove lo spread mangia il rendimento.",
    "prezzo_massimo": "Tetto di prezzo: serve solo a escludere anomalie di "
                      "quotazione.",
    "dollar_volume_minimo": "Prezzo per pezzi scambiati, media a novanta "
                            "giorni. Misura se il titolo e' davvero "
                            "negoziabile.",
    "tasso_privo_rischio": "Usato per lo Sharpe ratio, dei singoli titoli e "
                           "dei portafogli.",
}

RIBILANCIAMENTO_HELP = {
    "NONE": "I pesi restano uguali ogni giorno. Suona come 'non faccio "
            "niente', ma e' l'opposto: equivale a ribilanciare tutte le sere. "
            "E' il comportamento del notebook del corso, mantenuto per "
            "riprodurne i risultati.",
    "BUYHOLD": "Si comprano quote uguali il primo giorno e non si tocca piu' "
               "nulla. I pesi derivano: dopo vent'anni il titolo migliore puo' "
               "arrivare a pesare meta' del portafoglio. E' il vero 'non "
               "faccio niente'.",
    "QUARTERLY": "I pesi derivano dentro il trimestre e vengono riportati a "
                 "1/n l'ultimo giorno di borsa del trimestre.",
    "SEMIANNUALLY": "Come sopra, ma ogni sei mesi.",
    "ANNUALLY": "Come sopra, ma una volta l'anno.",
}

SIMULAZIONE_HELP = """
Il numero di simulazioni non e' il numero di portafogli possibili: e' il
numero di **estrazioni a sorte**. Con cento titoli promossi e portafogli da
cinque a dieci titoli, le combinazioni possibili sono qualche centinaio di
migliaia di miliardi. Cinquantamila estrazioni non le esplorano: ne
descrivono la forma, che e' quello che serve.

Aumentare le simulazioni migliora un po' i portafogli notevoli e non cambia
la nuvola. Il seme casuale serve a rendere ripetibile l'estrazione: con lo
stesso seme e gli stessi titoli si ottengono gli stessi portafogli, il che e'
indispensabile in aula.
"""

# ==========================================================================
# Metodologia
# ==========================================================================
METODOLOGIA = """
### Dati

Prezzi *adjusted close* giornalieri di EOD Historical Data, corretti per
dividendi e frazionamenti. L'universo e' la lista ufficiale dei simboli
quotati, ristretta ad **azioni ordinarie ed ETF**: restano fuori azioni
privilegiate, warrant, fondi chiusi e obbligazioni.

Tutti i titoli dell'universo vengono scaricati e misurati. Non c'e' alcun
pre-filtro: la selezione avviene dopo, sui dati completi.

### Misure dei singoli titoli

- **Anzianita'**: giorni fra la prima e l'ultima quotazione, diviso 365,25.
- **Volume in dollari**: media di prezzo x volume sugli ultimi 90 giorni di borsa.
- **Rendimenti giornalieri**: variazione semplice, saltando i giorni in cui il
  prezzo precedente non e' positivo.
- **Drawdown massimo**: minimo di (prezzo / massimo raggiunto fino a quel
  giorno - 1) su tutta la storia.
- **Sharpe ratio**: (media dei rendimenti x 252 - tasso privo di rischio)
  diviso (deviazione standard x radice di 252). Deviazione standard di
  popolazione, come nel notebook.
- **Volatilita' mensile annualizzata**: deviazione standard dei rendimenti
  mensili (ultimo prezzo di ogni mese), moltiplicata per la radice di 12.
- **Performance continua a N anni**: minimo di tutti i rendimenti fra il
  giorno t e il giorno t + N x 252.

### Portafogli

Pesi **uguali** per tutti i titoli. Il rendimento atteso e' la media dei
rendimenti attesi dei componenti; la varianza e' la somma dell'intera
sottomatrice di covarianza divisa per il quadrato del numero di titoli.
Media e covarianza sono annualizzate moltiplicando per 252, e calcolate sui
rendimenti giornalieri con i buchi azzerati.

L'estrazione dei portafogli e' uniforme: prima si sorteggia il numero di
titoli fra il minimo e il massimo indicati, poi il sottoinsieme di quella
dimensione.

### Serie storiche dei portafogli

Prima della simulazione vengono esclusi i titoli la cui storia comincia dopo
la data limite (anni richiesti meno sessanta giorni di tolleranza), perche'
un solo titolo troppo giovane accorcia la storia comune di qualunque
portafoglio lo contenga.

Per la ricostruzione delle serie si eliminano le giornate in cui manca il
prezzo di almeno uno dei componenti.

### Che cosa non e' modellato

**Costi.** Niente commissioni, niente spread denaro-lettera, niente impatto
di mercato. Su un portafoglio ribilanciato ogni giorno - che e'
l'impostazione predefinita - questa e' un'assunzione molto forte, e va
tenuta a mente prima di confrontare il ribilanciamento giornaliero con il
buy and hold.

**Fiscalita'.** Nessuna imposta su plusvalenze o dividendi. Il
ribilanciamento a vendita e' un evento fiscale ogni volta che l'asset
venduto e' in guadagno.

**Titoli usciti dal listino.** L'archivio contiene i simboli quotati **oggi**.
Le societa' fallite, fuse o ritirate non ci sono. E' la forma piu' classica
di *survivorship bias*, e da sola basta a spiegare una parte non piccola dei
rendimenti che si vedranno.

**Il fatto che la selezione guardi al passato.** I titoli sono scelti perche'
si sono comportati bene in un periodo, e poi misurati **su quello stesso
periodo**. E' il difetto piu' grave dell'esercizio, ed e' insanabile per
costruzione. L'unico modo serio di affrontarlo sarebbe selezionare sui primi
dieci anni e misurare sui secondi dieci.
"""

GLOSSARIO = [
    ("Adjusted close",
     "Prezzo di chiusura corretto per dividendi e frazionamenti. E' l'unica "
     "serie su cui abbia senso misurare un rendimento pluriennale, perche' "
     "il prezzo grezzo si dimezza il giorno di uno split senza che nessuno "
     "abbia perso nulla."),
    ("CAGR",
     "Crescita annua composta: il tasso costante che, applicato per tutto il "
     "periodo, porterebbe dal valore iniziale a quello finale."),
    ("Drawdown",
     "Distanza dal massimo precedente. Il drawdown massimo e' il punto piu' "
     "basso mai raggiunto rispetto a un massimo precedente."),
    ("Frontiera efficiente",
     "L'insieme dei portafogli per cui non esiste alternativa che renda di "
     "piu' a parita' di rischio, ne' che rischi meno a parita' di rendimento."),
    ("Monte Carlo",
     "Metodo che esplora un problema estraendo a sorte molte soluzioni "
     "possibili invece di enumerarle tutte. Qui: molte combinazioni di titoli."),
    ("Sharpe ratio",
     "Rendimento in eccesso rispetto al tasso privo di rischio, diviso per la "
     "volatilita'. Quanto si e' stati pagati per il rischio corso."),
    ("Survivorship bias",
     "La distorsione che nasce dall'osservare solo cio' che e' sopravvissuto. "
     "Un listino di oggi non contiene le societa' fallite ieri, e questo "
     "gonfia sistematicamente ogni statistica calcolata all'indietro."),
    ("Volatilita'",
     "Deviazione standard dei rendimenti, annualizzata. Misura di quanto un "
     "prezzo oscilli, non della direzione in cui va."),
    ("Volume in dollari",
     "Prezzo per numero di pezzi scambiati. Misura quanto denaro passa di "
     "mano ogni giorno, cioe' se il titolo e' davvero negoziabile."),
]
