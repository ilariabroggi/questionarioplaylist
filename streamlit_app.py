import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
import json
import base64
import requests

from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import quote_plus
from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# CONFIGURAZIONE DELLA PAGINA
# ============================================================

st.set_page_config(
    page_title="Playlist ed emozioni",
    page_icon="🎵",
    layout="centered"
)


# ============================================================
# INIZIALIZZAZIONE SESSION STATE
# ============================================================

if "analisi_completata" not in st.session_state:
    st.session_state.analisi_completata = False

if "playlist_generata" not in st.session_state:
    st.session_state.playlist_generata = False

if "final_emotion" not in st.session_state:
    st.session_state.final_emotion = None

if "music_mood" not in st.session_state:
    st.session_state.music_mood = None

if "playlist_df" not in st.session_state:
    st.session_state.playlist_df = None

if "playlist_fallback_used" not in st.session_state:
    st.session_state.playlist_fallback_used = False

if "questionario_aperto" not in st.session_state:
    st.session_state.questionario_aperto = False

if "risultato_salvato" not in st.session_state:
    st.session_state.risultato_salvato = False


# ============================================================
# COLLEGAMENTO A GITHUB PER I RISULTATI DEL QUESTIONARIO
# ============================================================

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]

GITHUB_OWNER = "ilariabroggi"
GITHUB_REPO = "questionarioplaylist"
GITHUB_BRANCH = "main"

GITHUB_RESULTS_FILE = "risultati_questionario.csv"

GITHUB_API_URL = (
    f"https://api.github.com/repos/"
    f"{GITHUB_OWNER}/"
    f"{GITHUB_REPO}/contents/"
    f"{GITHUB_RESULTS_FILE}"
)


# ============================================================
# COLONNE DEL FILE DEI RISULTATI
# ============================================================

RESULTS_COLUMNS = [
    "id_partecipante",
    "timestamp",
    "genere_preferito",
    "lingua_preferita",
    "anno_inizio",
    "anno_fine",
    "preferenza_popolarita",
    "emozione_dichiarata",
    "emozione_stimata",
    "mood",
    "numero_canzoni",
    "song_id_1",
    "song_id_2",
    "song_id_3",
    "song_id_4",
    "song_id_5",
    "song_id_6",
    "song_id_7",
    "gradimento_playlist",
    "coerenza_emotiva",
    "canzone_preferita",
    "canzone_meno_preferita",
    "canzoni_gia_conosciute",
    "frequenza_ascolto_musica",
    "commento",
    "genere",
    "eta",
    "regione_residenza",
    "istruzione"
]


# ============================================================
# LETTURA DEL FILE RISULTATI DA GITHUB
# ============================================================

def load_results_from_github():

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10"
    }

    response = requests.get(
        GITHUB_API_URL,
        headers=headers,
        params={
            "ref": GITHUB_BRANCH
        },
        timeout=30
    )

    if response.status_code != 200:

        raise RuntimeError(
            "Errore durante la lettura del file "
            "risultati_questionario.csv da GitHub.\n\n"
            f"Codice HTTP: {response.status_code}\n"
            f"Risposta GitHub: {response.text}"
        )

    data = response.json()

    file_sha = data["sha"]

    encoded_content = data["content"]

    decoded_content = base64.b64decode(
        encoded_content
    ).decode(
        "utf-8"
    )

    results_df = pd.read_csv(
        StringIO(decoded_content)
    )

    return results_df, file_sha


# ============================================================
# GENERAZIONE ID PARTECIPANTE
# ============================================================

def generate_participant_id(results_df):

    if results_df.empty:
        return "0001"

    if "id_partecipante" not in results_df.columns:
        return "0001"

    existing_ids = (
        results_df["id_partecipante"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
    )

    numeric_ids = pd.to_numeric(
        existing_ids,
        errors="coerce"
    )

    numeric_ids = numeric_ids.dropna()

    if numeric_ids.empty:
        next_id = 1
    else:
        next_id = int(
            numeric_ids.max()
        ) + 1

    return f"{next_id:04d}"


# ============================================================
# SALVATAGGIO DI UN NUOVO RISULTATO SU GITHUB
# ============================================================

def save_result_to_github(new_row):

    # ========================================================
    # PRIMO TENTATIVO
    # ========================================================

    for attempt in range(3):

        results_df, file_sha = load_results_from_github()

        # ----------------------------------------------------
        # CREIAMO UN DATAFRAME CON LA NUOVA RISPOSTA
        # ----------------------------------------------------

        new_row_df = pd.DataFrame(
            [new_row]
        )

        # ----------------------------------------------------
        # CI ASSICURIAMO CHE TUTTE LE COLONNE
        # SIANO PRESENTI
        # ----------------------------------------------------

        for column in RESULTS_COLUMNS:

            if column not in new_row_df.columns:

                new_row_df[column] = pd.NA

        # ----------------------------------------------------
        # MANTENIAMO ESATTAMENTE L'ORDINE
        # DELLE COLONNE
        # ----------------------------------------------------

        new_row_df = new_row_df[
            RESULTS_COLUMNS
        ]

        # ----------------------------------------------------
        # AGGIUNGIAMO LA NUOVA RIGA
        # ----------------------------------------------------

        updated_df = pd.concat(
            [
                results_df,
                new_row_df
            ],
            ignore_index=True
        )

        # ----------------------------------------------------
        # CONVERSIONE DEL DATAFRAME IN CSV
        # ----------------------------------------------------

        csv_content = updated_df.to_csv(
            index=False
        )

        # ----------------------------------------------------
        # CODIFICA BASE64
        # ----------------------------------------------------

        encoded_content = base64.b64encode(
            csv_content.encode("utf-8")
        ).decode("utf-8")

        # ----------------------------------------------------
        # HEADER DELLA RICHIESTA
        # ----------------------------------------------------

        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10"
        }

        # ----------------------------------------------------
        # DATI DA INVIARE A GITHUB
        # ----------------------------------------------------

        payload = {
            "message": (
                f"Salvataggio questionario "
                f"{new_row['id_partecipante']}"
            ),
            "content": encoded_content,
            "sha": file_sha,
            "branch": GITHUB_BRANCH
        }

        # ----------------------------------------------------
        # AGGIORNAMENTO DEL FILE
        # ----------------------------------------------------

        response = requests.put(
            GITHUB_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        # ----------------------------------------------------
        # SALVATAGGIO RIUSCITO
        # ----------------------------------------------------

        if response.status_code in [200, 201]:

            return True

        # ----------------------------------------------------
        # CONFLITTO DI VERSIONE
        #
        # Rileggiamo il file e riproviamo.
        # ----------------------------------------------------

        if response.status_code == 409:

            if attempt < 2:
                continue

            raise RuntimeError(
                "CONFLICT_GITHUB"
            )

        # ----------------------------------------------------
        # ALTRI ERRORI
        # ----------------------------------------------------

        raise RuntimeError(
            "Errore durante il salvataggio del risultato "
            "su GitHub.\n\n"
            f"Codice HTTP: {response.status_code}\n"
            f"Risposta GitHub: {response.text}"
        )

    return False


# ============================================================
# CARICAMENTO DEI FILE NECESSARI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

centroids = pd.read_csv(
    BASE_DIR / "centroidi_emozioni.csv",
    index_col=0
)

scaler = joblib.load(
    BASE_DIR / "scaler_mfcc.pkl"
)

with open(
    BASE_DIR / "feature_cols.json",
    "r"
) as file:
    feature_cols = json.load(file)


# ============================================================
# NORMALIZZAZIONE DEI NOMI DELLE EMOZIONI
# ============================================================

centroids.index = [
    str(emotion).strip().lower()
    for emotion in centroids.index
]


# ============================================================
# EMOZIONI AMMESSE PER IL RISULTATO FINALE
# ============================================================

final_emotions = [
    "calm",
    "happy",
    "sad",
    "angry"
]


missing_final_emotions = [
    emotion
    for emotion in final_emotions
    if emotion not in centroids.index
]

if missing_final_emotions:

    st.error(
        "Nel file centroidi_emozioni.csv non sono state trovate "
        f"queste emozioni necessarie per il risultato finale: "
        f"{', '.join(missing_final_emotions)}"
    )

    st.stop()


# ============================================================
# CONTROLLO DELLE FEATURE MFCC
# ============================================================

missing_features = [
    feature
    for feature in feature_cols
    if feature not in centroids.columns
]

if missing_features:

    st.error(
        "Nel file centroidi_emozioni.csv mancano alcune feature MFCC: "
        f"{', '.join(missing_features)}"
    )

    st.stop()


# ============================================================
# MAPPATURA EMOZIONE VOCALE → MOOD MUSICALE
# ============================================================

emotion_to_music_mood = {
    "calm": "calm",
    "happy": "happy",
    "sad": "sad",
    "angry": "energetic"
}


# ============================================================
# FILE DEI DATASET MUSICALI
# ============================================================

music_dataset_files = {
    "calm": "songs_clean_calm.csv",
    "happy": "songs_clean_happy.csv",
    "sad": "songs_clean_sad.csv",
    "energetic": "songs_clean_energetic.csv.gz"
}


# ============================================================
# CONVERSIONE DEI VALORI DEL QUESTIONARIO
# ============================================================

genre_mapping = {
    "Rock": "Rock",
    "Pop": "Pop",
    "Elettronica": "Electronic",
    "Folk": "Folk",
    "Country": "Country",
    "Hip-Hop": "Hip-Hop",
    "Rhythm and Blues": "R&B",
    "Jazz": "Jazz",
    "Blues": "Blues",
    "Colonne sonore e musica classica": "Classical"
}


language_mapping = {
    "Italiano": "italian",
    "Inglese": "english",
    "Spagnolo": "spanish",
    "Portoghese": "portuguese",
    "Francese": "french",
    "Tedesco": "german"
}


# ============================================================
# FUNZIONE PER ESTRARRE LE FEATURE MFCC
# ============================================================

def extract_mfcc_features(audio_file):

    audio_bytes = audio_file.getvalue()

    signal, sample_rate = librosa.load(
        BytesIO(audio_bytes),
        sr=48000
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    feature_vector = np.concatenate(
        [
            mfcc_mean,
            mfcc_std
        ]
    )

    features_df = pd.DataFrame(
        [feature_vector],
        columns=feature_cols
    )

    return features_df


# ============================================================
# FUNZIONE PER CALCOLARE LE DISTANZE DAI CENTROIDI
# ============================================================

def calculate_distances(features_df):

    features_std = scaler.transform(
        features_df[feature_cols]
    )

    features_std_df = pd.DataFrame(
        features_std,
        columns=feature_cols
    )

    point = features_std_df.iloc[0].values

    distances = {}

    for emotion in centroids.index:

        centroid = centroids.loc[
            emotion,
            feature_cols
        ].values.astype(float)

        distance = np.linalg.norm(
            point - centroid
        )

        distances[emotion] = distance

    return features_std_df, distances


# ============================================================
# FUNZIONE PER CLASSIFICARE L'EMOZIONE
# ============================================================

def predict_emotion(features_df):

    features_std_df, distances = calculate_distances(
        features_df
    )

    if not distances:

        raise ValueError(
            "Non è stato possibile calcolare le distanze "
            "dai centroidi."
        )

    predicted_emotion = min(
        distances,
        key=distances.get
    )

    return (
        predicted_emotion,
        distances,
        features_std_df
    )


# ============================================================
# CARICAMENTO DEL DATASET MUSICALE
# ============================================================

@st.cache_data
def load_music_dataset(music_mood):

    if music_mood not in music_dataset_files:

        raise ValueError(
            f"Mood musicale non riconosciuto: {music_mood}"
        )

    file_name = music_dataset_files[music_mood]

    file_path = BASE_DIR / file_name

    if not file_path.exists():

        raise FileNotFoundError(
            f"Non è stato trovato il dataset musicale: {file_name}"
        )

    required_columns = [
        "id",
        "name",
        "album_name",
        "artists",
        "year",
        "genre",
        "mood",
        "language",
        "mood_distance",
        "popularity"
    ]

    header = pd.read_csv(
        file_path,
        nrows=0
    )

    actual_columns = list(
        header.columns
    )

    column_map = {
        str(column).strip().lower(): column
        for column in actual_columns
    }

    missing_columns = [
        column
        for column in required_columns
        if column not in column_map
    ]

    if missing_columns:

        raise ValueError(
            "Nel dataset "
            f"{file_name} "
            "mancano le seguenti colonne necessarie: "
            +
            ", ".join(missing_columns)
            +
            ".\n\n"
            "Colonne effettivamente presenti nel file: "
            +
            ", ".join(
                str(column)
                for column in actual_columns
            )
        )

    actual_required_columns = [
        column_map[column]
        for column in required_columns
    ]

    df = pd.read_csv(
        file_path,
        usecols=actual_required_columns
    )

    rename_map = {
        actual_column: normalized_column
        for normalized_column, actual_column
        in zip(
            required_columns,
            actual_required_columns
        )
    }

    df = df.rename(
        columns=rename_map
    )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    df["mood_distance"] = pd.to_numeric(
        df["mood_distance"],
        errors="coerce"
    )

    df["popularity"] = pd.to_numeric(
        df["popularity"],
        errors="coerce"
    )

    df["language"] = (
        df["language"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["genre"] = (
        df["genre"]
        .astype("string")
        .str.strip()
    )

    df = df[
        df["year"].notna()
        &
        df["mood_distance"].notna()
        &
        df["popularity"].notna()
    ].copy()

    return df


# ============================================================
# ORDINAMENTO DI UN BLOCCO DI CANDIDATE
# ============================================================

def rank_block(
    df,
    popularity_preference
):

    if df.empty:

        return df.copy()

    if popularity_preference == "Canzoni popolari":

        return (
            df
            .sort_values(
                by="popularity",
                ascending=False
            )
            .copy()
        )

    else:

        return (
            df
            .sort_values(
                by="mood_distance",
                ascending=True
            )
            .copy()
        )


# ============================================================
# GENERAZIONE DELLA PLAYLIST
# ============================================================

def generate_playlist(
    music_df,
    preferred_language,
    start_year,
    end_year,
    preferred_genre,
    popularity_preference,
    target_songs=7
):

    music_language = language_mapping[
        preferred_language
    ]

    music_genre = genre_mapping[
        preferred_genre
    ]

    selected_blocks = []

    selected_ids = set()

    def current_total():

        return sum(
            len(block)
            for block in selected_blocks
        )

    def add_block(block_df):

        nonlocal selected_blocks
        nonlocal selected_ids

        if block_df.empty:

            return 0

        # ----------------------------------------------------
        # EVITIAMO DI INSERIRE DUE VOLTE LA STESSA CANZONE
        # ----------------------------------------------------

        block_df = block_df[
            ~block_df["id"]
            .astype(str)
            .isin(selected_ids)
        ].copy()

        if block_df.empty:

            return 0

        # ----------------------------------------------------
        # OGNI BLOCCO VIENE ORDINATO SEPARATAMENTE
        # ----------------------------------------------------

        block_df = rank_block(
            block_df,
            popularity_preference
        )

        # ----------------------------------------------------
        # CALCOLIAMO QUANTI POSTI SONO ANCORA DISPONIBILI
        # ----------------------------------------------------

        remaining_slots = (
            target_songs
            -
            current_total()
        )

        if remaining_slots <= 0:

            return 0

        # ----------------------------------------------------
        # PRENDIAMO SOLO LE CANZONI NECESSARIE
        # ----------------------------------------------------

        block_df = block_df.head(
            remaining_slots
        ).copy()

        # ----------------------------------------------------
        # SALVIAMO GLI ID GIÀ UTILIZZATI
        # ----------------------------------------------------

        selected_ids.update(
            block_df["id"].astype(str)
        )

        # ----------------------------------------------------
        # MANTENIAMO IL BLOCCO SEPARATO
        # ----------------------------------------------------
        #
        # Questo è importante:
        # le canzoni di un blocco successivo NON vengono
        # rimescolate insieme alle canzoni dei blocchi precedenti.
        #

        selected_blocks.append(
            block_df
        )

        return len(block_df)

    # ========================================================
    # BLOCCO 1
    # MOOD + GENERE + PERIODO + LINGUA
    # ========================================================
    #
    # Il mood è già garantito dal dataset musicale caricato.
    #
    # Questo è il livello che rispetta TUTTE le preferenze.
    # ========================================================

    block_1 = music_df[
        (music_df["genre"] == music_genre)
        &
        (music_df["year"] >= start_year)
        &
        (music_df["year"] <= end_year)
        &
        (music_df["language"] == music_language)
    ].copy()

    full_criteria_count = len(block_1)

    add_block(
        block_1
    )

    # ========================================================
    # BLOCCO 2
    # MOOD + GENERE + PERIODO
    # ========================================================
    #
    # Se non sono bastate le canzoni del primo blocco,
    # eliminiamo SOLO il criterio della lingua.
    #
    # Genere e periodo rimangono obbligatori.
    # ========================================================

    if current_total() < target_songs:

        block_2 = music_df[
            (music_df["genre"] == music_genre)
            &
            (music_df["year"] >= start_year)
            &
            (music_df["year"] <= end_year)
        ].copy()

        add_block(
            block_2
        )

    # ========================================================
    # BLOCCO 3
    # MOOD + GENERE
    # ========================================================
    #
    # Se non sono ancora bastate 7 canzoni,
    # eliminiamo anche il criterio del periodo.
    #
    # Il genere rimane obbligatorio.
    # ========================================================

    if current_total() < target_songs:

        block_3 = music_df[
            music_df["genre"] == music_genre
        ].copy()

        add_block(
            block_3
        )

    # ========================================================
    # BLOCCO 4
    # MOOD
    # ========================================================
    #
    # Se non sono ancora bastate 7 canzoni,
    # eliminiamo anche il genere.
    #
    # Rimane SEMPRE il mood determinato dalla voce.
    # ========================================================

    if current_total() < target_songs:

        block_4 = music_df.copy()

        add_block(
            block_4
        )

    # ========================================================
    # RISULTATO FINALE
    # ========================================================

    if not selected_blocks:

        playlist = pd.DataFrame(
            columns=music_df.columns
        )

    else:

        playlist = pd.concat(
            selected_blocks,
            ignore_index=True
        )

    # ========================================================
    # VERIFICA SE È STATO NECESSARIO RILASSARE I CRITERI
    # ========================================================
    #
    # Il messaggio viene mostrato solamente se non erano
    # disponibili almeno 7 canzoni che rispettassero
    # contemporaneamente:
    #
    # mood + genere + periodo + lingua
    #
    # Non importa quale blocco sia poi servito per completare
    # la playlist.
    # ========================================================

    fallback_used = (
        full_criteria_count < target_songs
    )

    return (
        playlist,
        fallback_used
    )


# ============================================================
# CREAZIONE LINK SPOTIFY
# ============================================================

def create_spotify_link(song_id):

    song_id = str(
        song_id
    ).strip()

    return (
        f"https://open.spotify.com/track/{song_id}"
    )


# ============================================================
# CREAZIONE LINK YOUTUBE
# ============================================================

def create_youtube_link(
    title,
    artist
):

    query = quote_plus(
        f"{artist} {title}"
    )

    return (
        "https://www.youtube.com/results?search_query="
        +
        query
    )


# ============================================================
# TITOLO
# ============================================================

st.title(
    "🎵 Emozioni, voce e musica"
)

st.subheader(
    "Un progetto di tesi sulla relazione tra "
    "emozioni vocali e preferenze musicali"
)


# ============================================================
# INTRODUZIONE
# ============================================================

st.write("""
Ciao! 👋

Sono una studentessa dell'Università degli Studi di Modena e Reggio Emilia
e sto realizzando questo progetto nell'ambito della mia tesi di laurea
nel corso di studi in Pubblicità, Comunicazione Digitale e Creatività d'Impresa.

Questa pagina web è stata realizzata per esplorare la relazione tra le emozioni
espresse attraverso la voce e le preferenze musicali.

Durante questa esperienza ti verranno poste alcune brevi domande sulle tue
preferenze musicali. Successivamente ti sarà richiesto di registrare la tua voce
pronunciando alcune semplici frasi.

La registrazione verrà analizzata per stimare quale emozione risulta maggiormente
espressa attraverso le caratteristiche acustiche della tua voce.

In base al risultato ottenuto e alle preferenze musicali indicate, ti verrà
successivamente proposta una playlist musicale.

Al termine dell'ascolto potrai inoltre compilare un breve questionario finale
per valutare la playlist ricevuta e fornire alcune informazioni utili
all'analisi dei risultati del progetto.
""")


st.divider()


# ============================================================
# QUESTIONARIO SULLE PREFERENZE MUSICALI
# ============================================================

st.header(
    "🎧 Le tue preferenze musicali"
)

st.write("""
Prima di procedere con la registrazione della voce, rispondi alle seguenti
domande sulle tue preferenze musicali.
""")


# ============================================================
# GENERE MUSICALE
# ============================================================

genere = st.selectbox(
    "Qual è il tuo genere musicale preferito?",
    [
        "Rock",
        "Pop",
        "Elettronica",
        "Folk",
        "Country",
        "Hip-Hop",
        "Rhythm and Blues",
        "Jazz",
        "Blues",
        "Colonne sonore e musica classica"
    ]
)


# ============================================================
# LINGUA
# ============================================================

lingua = st.selectbox(
    "In quale lingua preferisci ascoltare musica?",
    [
        "Italiano",
        "Inglese",
        "Spagnolo",
        "Portoghese",
        "Francese",
        "Tedesco"
    ]
)


# ============================================================
# PERIODO MUSICALE
# ============================================================

years = list(
    range(
        1970,
        2026
    )
)

anno_inizio = st.selectbox(
    "Da quale anno vuoi ascoltare musica?",
    years,
    index=years.index(2000)
)

anno_fine = st.selectbox(
    "Fino a quale anno vuoi ascoltare musica?",
    years,
    index=years.index(2002)
)

if anno_fine < anno_inizio:

    st.warning(
        "⚠️ L'anno finale deve essere uguale o successivo "
        "all'anno iniziale."
    )


# ============================================================
# POPOLARITÀ DELLE CANZONI
# ============================================================

popularity_preference = st.radio(
    "Preferisci ascoltare canzoni popolari o canzoni poco conosciute?",
    [
        "Canzoni popolari",
        "Canzoni poco conosciute"
    ]
)


st.divider()


# ============================================================
# EMOZIONE DICHIARATA
# ============================================================

st.header(
    "💭 Come ti senti in questo momento?"
)

st.write("""
Scegli l'emozione o lo stato d'animo che ritieni maggiormente
rappresentativo di come ti senti in questo momento.

Questa risposta verrà utilizzata anche per confrontare la tua percezione
personale con l'emozione stimata successivamente attraverso l'analisi
della voce.
""")


emozione_dichiarata = st.radio(
    "Come descriveresti il tuo stato emotivo attuale?",
    [
        "Neutro",
        "Calmo",
        "Felice",
        "Triste",
        "Arrabbiato/Energetico",
        "Impaurito",
        "Disgustato",
        "Sorpreso"
    ]
)


st.divider()


# ============================================================
# REGISTRAZIONE DELLA VOCE
# ============================================================

st.header(
    "🎙️ Registrazione della voce"
)

st.write("""
Ora ti verrà chiesto di pronunciare due semplici frasi.

Pronuncia ciascuna frase cercando di far trasparire attraverso la voce
il tuo stato emotivo attuale.

Non cercare di interpretare artificialmente un'emozione:
pronuncia le frasi nel modo più naturale possibile, lasciando emergere
il tuo stato d'animo attraverso il tono della voce,
l'intonazione e l'intensità.
""")


# ============================================================
# PRIMA REGISTRAZIONE
# ============================================================

st.subheader(
    "Registrazione 1"
)

st.write(
    "Pronuncia la seguente frase:"
)

st.info(
    "🗣️ «I bambini stanno parlando vicino alla porta.»"
)

audio_1 = st.audio_input(
    "Registra la prima frase",
    sample_rate=48000,
    key="audio_1"
)


# ============================================================
# SECONDA REGISTRAZIONE
# ============================================================

st.subheader(
    "Registrazione 2"
)

st.write(
    "Pronuncia la seguente frase:"
)

st.info(
    "🗣️ «I cani sono seduti vicino alla porta.»"
)

audio_2 = st.audio_input(
    "Registra la seconda frase",
    sample_rate=48000,
    key="audio_2"
)


st.divider()


# ============================================================
# RIEPILOGO
# ============================================================

st.header(
    "📋 Riepilogo"
)

st.write(
    "Prima di procedere, verifica le informazioni inserite."
)

st.write(
    f"**Genere musicale preferito:** {genere}"
)

st.write(
    f"**Lingua preferita:** {lingua}"
)

st.write(
    f"**Periodo musicale preferito:** "
    f"{anno_inizio}–{anno_fine}"
)

st.write(
    f"**Preferenza di popolarità:** {popularity_preference}"
)

st.write(
    f"**Emozione dichiarata:** {emozione_dichiarata}"
)


# ============================================================
# ANALISI
# ============================================================

st.divider()

procedi = st.button(
    "Analizza la mia voce 🎵"
)


if procedi:

    if anno_fine < anno_inizio:

        st.error(
            "❌ Seleziona un periodo valido: "
            "l'anno finale deve essere uguale o successivo "
            "all'anno iniziale."
        )

    elif audio_1 is None or audio_2 is None:

        st.warning(
            "⚠️ Per continuare è necessario registrare entrambe le frasi."
        )

    else:

        try:

            with st.spinner(
                "Analisi della voce in corso..."
            ):

                # ====================================================
                # ESTRAZIONE DELLE FEATURE
                # ====================================================

                features_1 = extract_mfcc_features(
                    audio_1
                )

                features_2 = extract_mfcc_features(
                    audio_2
                )

                # ====================================================
                # STANDARDIZZAZIONE
                # ====================================================

                features_std_1 = scaler.transform(
                    features_1[feature_cols]
                )

                features_std_2 = scaler.transform(
                    features_2[feature_cols]
                )

                # ====================================================
                # EMOZIONE DELLA PRIMA REGISTRAZIONE
                # ====================================================

                point_1 = features_std_1[0]

                distances_1 = {}

                for emotion in centroids.index:

                    centroid = centroids.loc[
                        emotion,
                        feature_cols
                    ].values.astype(float)

                    distance = np.linalg.norm(
                        point_1 - centroid
                    )

                    distances_1[emotion] = distance

                if not distances_1:

                    raise ValueError(
                        "Non sono state calcolate distanze "
                        "per la prima registrazione."
                    )

                emotion_1 = min(
                    distances_1,
                    key=distances_1.get
                )

                # ====================================================
                # EMOZIONE DELLA SECONDA REGISTRAZIONE
                # ====================================================

                point_2 = features_std_2[0]

                distances_2 = {}

                for emotion in centroids.index:

                    centroid = centroids.loc[
                        emotion,
                        feature_cols
                    ].values.astype(float)

                    distance = np.linalg.norm(
                        point_2 - centroid
                    )

                    distances_2[emotion] = distance

                if not distances_2:

                    raise ValueError(
                        "Non sono state calcolate distanze "
                        "per la seconda registrazione."
                    )

                emotion_2 = min(
                    distances_2,
                    key=distances_2.get
                )

                # ====================================================
                # CREAZIONE DEL PUNTO MEDIO
                # ====================================================

                mean_point = (
                    features_std_1[0]
                    +
                    features_std_2[0]
                ) / 2

                # ====================================================
                # DISTANZE DEL PUNTO MEDIO
                # ====================================================

                mean_distances = {}

                for emotion in final_emotions:

                    centroid = centroids.loc[
                        emotion,
                        feature_cols
                    ].values.astype(float)

                    distance = np.linalg.norm(
                        mean_point - centroid
                    )

                    mean_distances[emotion] = distance

                if not mean_distances:

                    raise ValueError(
                        "Non sono state calcolate le distanze "
                        "per le quattro emozioni finali."
                    )

                # ====================================================
                # EMOZIONE FINALE
                # ====================================================

                final_emotion = min(
                    mean_distances,
                    key=mean_distances.get
                )

                # Salviamo il risultato nella sessione.

                st.session_state.final_emotion = final_emotion

            # ========================================================
            # GENERAZIONE DELLA PLAYLIST
            # ========================================================

            with st.spinner(
                "Sto cercando le canzoni più adatte a te..."
            ):

                music_mood = emotion_to_music_mood[
                    final_emotion
                ]

                music_df = load_music_dataset(
                    music_mood
                )

                playlist_df, fallback_used = generate_playlist(
                    music_df=music_df,
                    preferred_language=lingua,
                    start_year=anno_inizio,
                    end_year=anno_fine,
                    preferred_genre=genere,
                    popularity_preference=popularity_preference,
                    target_songs=7
                )

                # Salviamo playlist, mood e informazione
                # sull'eventuale rilassamento dei criteri.

                st.session_state.playlist_df = playlist_df

                st.session_state.playlist_fallback_used = (
                    fallback_used
                )

                st.session_state.music_mood = music_mood

                st.session_state.analisi_completata = True

                st.session_state.playlist_generata = True

                st.session_state.questionario_aperto = False

        except Exception as e:

            st.error(
                "❌ Si è verificato un errore durante l'analisi."
            )

            st.exception(e)


# ============================================================
# RISULTATO E PLAYLIST
# ============================================================

if (
    st.session_state.analisi_completata
    and
    st.session_state.playlist_generata
):

    final_emotion = st.session_state.final_emotion

    music_mood = st.session_state.music_mood

    playlist_df = st.session_state.playlist_df

    # ========================================================
    # RISULTATO EMOZIONE
    # ========================================================

    st.divider()

    st.success(
        "Analisi completata! 🎉"
    )

    st.header(
        "🧠 Emozione stimata"
    )

    st.subheader(
        final_emotion.capitalize()
    )

    # ========================================================
    # CONFRONTO CON EMOZIONE DICHIARATA
    # ========================================================

    declared_emotion_mapping = {
        "Neutro": "neutral",
        "Calmo": "calm",
        "Felice": "happy",
        "Triste": "sad",
        "Arrabbiato/Energetico": "angry",
        "Impaurito": "fearful",
        "Disgustato": "disgust",
        "Sorpreso": "surprised"
    }

    declared_emotion_normalized = (
        declared_emotion_mapping[
            emozione_dichiarata
        ]
    )

    if final_emotion != declared_emotion_normalized:

        st.info(
            f"Hai indicato **{emozione_dichiarata}**, "
            f"mentre l'analisi della voce ha stimato "
            f"**{final_emotion.capitalize()}**."
        )

    else:

        st.info(
            f"Hai indicato **{emozione_dichiarata}** "
            "e l'analisi della voce ha stimato "
            f"**{final_emotion.capitalize()}**."
        )

    # ========================================================
    # VISUALIZZAZIONE PLAYLIST
    # ========================================================

    st.divider()

    st.header(
        "🎵 La tua playlist"
    )

    if playlist_df.empty:

        st.warning(
            "Non sono state trovate canzoni compatibili "
            "con i criteri selezionati."
        )

    else:

        # ====================================================
        # MESSAGGIO DI ADATTAMENTO DEI CRITERI
        # ====================================================
        #
        # Compare solamente se non erano disponibili almeno
        # 7 canzoni che rispettassero contemporaneamente
        # genere + periodo + lingua + mood.
        # ====================================================

        if st.session_state.playlist_fallback_used:

            st.info(
                "ℹ️ Non tutte le canzoni della playlist soddisfano "
                "tutte le preferenze che hai indicato. "
                "Abbiamo quindi adattato progressivamente i criteri "
                "per trovare le canzoni più adatte tra quelle "
                "disponibili nel dataset. 🎵"
            )

        st.write(
            f"Sono state selezionate "
            f"**{len(playlist_df)} canzoni** "
            f"coerenti con il tuo stato emotivo "
            "e le tue preferenze."
        )

        for position, (_, song) in enumerate(
            playlist_df.iterrows(),
            start=1
        ):

            spotify_url = create_spotify_link(
                song["id"]
            )

            youtube_url = create_youtube_link(
                song["name"],
                song["artists"]
            )

            st.subheader(
                f"{position}. {song['name']}"
            )

            st.write(
                f"**Artista:** {song['artists']}"
            )

            st.write(
                f"**Album:** {song['album_name']}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.link_button(
                    "Spotify",
                    spotify_url,
                    use_container_width=True
                )

            with col2:

                st.link_button(
                    "YouTube",
                    youtube_url,
                    use_container_width=True
                )


        # ====================================================
        # QUESTIONARIO FINALE
        # ====================================================

        st.divider()

        st.header(
            "🎧 Hai finito di ascoltare la playlist?"
        )

        st.write(
            "Quando hai finito di ascoltare le canzoni, "
            "clicca sul pulsante qui sotto per rispondere "
            "a poche domande sulla playlist."
        )

        apri_questionario = st.button(
            "Ho finito di ascoltare 🎶",
            use_container_width=True
        )

        if apri_questionario:

            st.session_state.questionario_aperto = True


# ============================================================
# QUESTIONARIO FINALE
# ============================================================

if (
    st.session_state.analisi_completata
    and
    st.session_state.playlist_generata
    and
    st.session_state.questionario_aperto
    and
    not st.session_state.risultato_salvato
):

    playlist_df = st.session_state.playlist_df

    final_emotion = st.session_state.final_emotion

    music_mood = st.session_state.music_mood

    st.divider()

    st.header(
        "📝 Questionario finale"
    )

    st.write(
        "Rispondi alle seguenti domande pensando alla playlist "
        "che hai appena ascoltato."
    )

    # ========================================================
    # CREAZIONE DELLE ETICHETTE DELLE CANZONI
    # ========================================================

    song_options = []

    for position, (_, song) in enumerate(
        playlist_df.iterrows(),
        start=1
    ):

        song_title = str(
            song["name"]
        ).strip()

        song_artist = str(
            song["artists"]
        ).strip()

        song_id = str(
            song["id"]
        ).strip()

        # ----------------------------------------------------
        # AGGIUNGIAMO IL NUMERO DELLA POSIZIONE NELLA PLAYLIST
        #
        # Esempio:
        # 1 — Titolo — Artista
        # 2 — Titolo — Artista
        # ...
        # 7 — Titolo — Artista
        # ----------------------------------------------------

        song_label = (
            f"{position} — {song_title} — {song_artist}"
        )

        song_options.append(
            {
                "label": song_label,
                "id": song_id
            }
        )

    song_labels = [
        song["label"]
        for song in song_options
    ]

    # ========================================================
    # DOMANDA 1
    # ========================================================

    st.subheader(
        "1. Quanto ti è piaciuta la playlist?"
    )

    gradimento_playlist = st.radio(
        "Seleziona una risposta:",
        [
            "Per niente",
            "Poco",
            "Moderatamente",
            "Molto",
            "Moltissimo"
        ],
        key="gradimento_playlist"
    )

    # ========================================================
    # DOMANDA 2
    # ========================================================

    st.subheader(
        "2. La playlist era coerente con il tuo stato emotivo?"
    )

    coerenza_emotiva = st.radio(
        "Seleziona una risposta:",
        [
            "Per niente",
            "Poco",
            "Moderatamente",
            "Molto",
            "Moltissimo"
        ],
        key="coerenza_emotiva"
    )

    # ========================================================
    # DOMANDA 3
    # ========================================================

    st.subheader(
        "3. Quale canzone ti è piaciuta di più?"
    )

    canzone_preferita = st.radio(
        "Seleziona una canzone:",
        song_labels,
        key="canzone_preferita"
    )

    # ========================================================
    # DOMANDA 4
    # ========================================================

    st.subheader(
        "4. Quale canzone ti è piaciuta di meno?"
    )

    canzone_meno_preferita = st.radio(
        "Seleziona una canzone:",
        song_labels,
        key="canzone_meno_preferita"
    )

    # ========================================================
    # DOMANDA 5
    # ========================================================

    st.subheader(
        "5. Quante delle canzoni proposte dalla playlist "
        "già conoscevi?"
    )

    numero_canzoni_playlist = len(
        playlist_df
    )

    if numero_canzoni_playlist == 1:

        known_options = [
            "Nessuna",
            "Tutte"
        ]

    else:

        known_options = (
            ["Nessuna"]
            +
            [
                str(number)
                for number in range(
                    1,
                    numero_canzoni_playlist
                )
            ]
            +
            ["Tutte"]
        )

    canzoni_gia_conosciute = st.radio(
        "Seleziona una risposta:",
        known_options,
        key="canzoni_gia_conosciute"
    )

    # ========================================================
    # DOMANDA 6
    # ========================================================

    st.subheader(
        "6. Quanto spesso ascolti musica al giorno?"
    )

    frequenza_ascolto_musica = st.radio(
        "Seleziona una risposta:",
        [
            "Meno di 30 minuti",
            "Da 30 minuti a meno di 1 ora",
            "Da 1 ora a meno di 2 ore",
            "Da 2 ore a meno di 3 ore",
            "3 ore o più"
        ],
        key="frequenza_ascolto_musica"
    )

    # ========================================================
    # DOMANDA 7
    # ========================================================

    st.subheader(
        "7. Se c'è qualcosa che vuoi dire, faccelo sapere."
    )

    commento = st.text_area(
        "Commento",
        placeholder=(
            "Scrivi qui eventuali osservazioni, "
            "commenti o suggerimenti..."
        ),
        key="commento"
    )

    # ========================================================
    # DOMANDE SOCIO-DEMOGRAFICHE
    # ========================================================

    st.divider()

    st.header(
        "👤 Domande socio-demografiche"
    )

    # ========================================================
    # GENERE
    # ========================================================

    genere_options = [
        "Maschio",
        "Femmina",
        "Altro"
    ]

    genere_demografico = st.radio(
        "Indica il tuo genere:",
        genere_options,
        key="genere_demografico"
    )

    # ========================================================
    # ETÀ
    # ========================================================

    age_options = (
        ["Meno di 14 anni"]
        +
        [
            str(age)
            for age in range(
                14,
                81
            )
        ]
        +
        ["80 anni o più"]
    )

    eta = st.selectbox(
        "Indica la tua età:",
        age_options,
        key="eta"
    )

    # ========================================================
    # REGIONE DI RESIDENZA
    # ========================================================

    region_options = [
        "Abruzzo",
        "Basilicata",
        "Calabria",
        "Campania",
        "Emilia-Romagna",
        "Friuli-Venezia Giulia",
        "Lazio",
        "Liguria",
        "Lombardia",
        "Marche",
        "Molise",
        "Piemonte",
        "Puglia",
        "Sardegna",
        "Sicilia",
        "Toscana",
        "Trentino-Alto Adige",
        "Umbria",
        "Valle d'Aosta",
        "Veneto",
        "Vivo all'estero"
    ]

    regione_residenza = st.selectbox(
        "Indica la tua regione di residenza:",
        region_options,
        key="regione_residenza"
    )

    # ========================================================
    # ISTRUZIONE
    # ========================================================

    istruzione_options = [
        "Terza media",
        "Diploma scuola superiore",
        "Laurea triennale",
        "Laurea magistrale a ciclo unico",
        "Dottorato di ricerca"
    ]

    istruzione = st.selectbox(
        "Indica il tuo titolo di studio:",
        istruzione_options,
        key="istruzione"
    )

    # ========================================================
    # INVIO QUESTIONARIO
    # ========================================================

    st.divider()

    st.write(
        "Quando hai completato tutte le domande, "
        "clicca sul pulsante qui sotto per inviare "
        "le tue risposte."
    )

    invia_questionario = st.button(
        "Invia il questionario e termina 🎵",
        type="primary",
        use_container_width=True
    )

    if invia_questionario:

        try:

            with st.spinner(
                "Salvataggio delle risposte in corso..."
            ):

                # ====================================================
                # LETTURA DEL FILE ATTUALE
                # ====================================================

                results_df, _ = load_results_from_github()

                # ====================================================
                # GENERAZIONE ID PARTECIPANTE
                # ====================================================

                participant_id = generate_participant_id(
                    results_df
                )

                # ====================================================
                # TIMESTAMP
                # ====================================================

                timestamp = datetime.now(
                    ZoneInfo("Europe/Rome")
                ).isoformat(
                    timespec="seconds"
                )

                # ====================================================
                # ID DELLE CANZONI
                #
                # Vengono presi direttamente dalla colonna "id"
                # del dataset musicale.
                # ====================================================

                song_ids = [
                    str(song["id"]).strip()
                    for _, song
                    in playlist_df.iterrows()
                ]

                # ====================================================
                # PREPARIAMO song_id_1 ... song_id_7
                # ====================================================

                song_id_columns = {}

                for index in range(7):

                    column_name = (
                        f"song_id_{index + 1}"
                    )

                    if index < len(song_ids):

                        song_id_columns[
                            column_name
                        ] = song_ids[index]

                    else:

                        song_id_columns[
                            column_name
                        ] = pd.NA

                # ====================================================
                # CREAZIONE DELLA NUOVA RIGA
                # ====================================================

                new_row = {

                    "id_partecipante":
                        participant_id,

                    "timestamp":
                        timestamp,

                    "genere_preferito":
                        genere,

                    "lingua_preferita":
                        lingua,

                    "anno_inizio":
                        anno_inizio,

                    "anno_fine":
                        anno_fine,

                    "preferenza_popolarita":
                        popularity_preference,

                    "emozione_dichiarata":
                        emozione_dichiarata,

                    "emozione_stimata":
                        final_emotion,

                    "mood":
                        music_mood,

                    "numero_canzoni":
                        len(playlist_df),

                    "gradimento_playlist":
                        gradimento_playlist,

                    "coerenza_emotiva":
                        coerenza_emotiva,

                    "canzone_preferita":
                        canzone_preferita,

                    "canzone_meno_preferita":
                        canzone_meno_preferita,

                    "canzoni_gia_conosciute":
                        canzoni_gia_conosciute,

                    "frequenza_ascolto_musica":
                        frequenza_ascolto_musica,

                    "commento":
                        commento,

                    "genere":
                        genere_demografico,

                    "eta":
                        eta,

                    "regione_residenza":
                        regione_residenza,

                    "istruzione":
                        istruzione
                }

                # ====================================================
                # AGGIUNGIAMO GLI ID DELLE CANZONI
                # ====================================================

                new_row.update(
                    song_id_columns
                )

                # ====================================================
                # SALVIAMO SU GITHUB
                # ====================================================

                save_result_to_github(
                    new_row
                )

                # ====================================================
                # SESSIONE COMPLETATA
                # ====================================================

                st.session_state.risultato_salvato = True

            st.success(
                "✅ Grazie! Le tue risposte sono state "
                "registrate correttamente."
            )

            st.write(
                "Puoi chiudere questa pagina."
            )

        except RuntimeError as e:

            if str(e) == "CONFLICT_GITHUB":

                st.error(
                    "⚠️ Si è verificato un conflitto durante "
                    "il salvataggio dei dati su GitHub. "
                    "Riprova tra qualche secondo."
                )

            else:

                st.error(
                    "❌ Si è verificato un errore durante "
                    "il salvataggio del questionario."
                )

                st.exception(e)

        except Exception as e:

            st.error(
                "❌ Si è verificato un errore durante "
                "il salvataggio del questionario."
            )

            st.exception(e)


# ============================================================
# MESSAGGIO FINALE DOPO IL SALVATAGGIO
# ============================================================

if st.session_state.risultato_salvato:

    st.divider()

    st.success(
        "🎉 Questionario completato! "
        "Grazie per aver partecipato al progetto."
    )