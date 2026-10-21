import base64
import json
import re
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import joblib
import librosa
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIGURAZIONE DELLA PAGINA
# ============================================================

st.set_page_config(
    page_title="Playlist ed emozioni",
    page_icon="🎵",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent

# Le etichette interne restano in inglese, perché devono coincidere
# con i nomi dei centroidi SER e con i nomi dei file musicali.
EMOTIONS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

EMOTION_ITALIAN = {
    "happy": "Felice",
    "surprised": "Sorpreso",
    "calm": "Calmo",
    "neutral": "Neutro",
    "angry": "Arrabbiato",
    "fearful": "Impaurito",
    "disgust": "Disgustato",
    "sad": "Triste",
}

DECLARED_EMOTION_MAPPING = {
    "Neutro": "neutral",
    "Calmo": "calm",
    "Felice": "happy",
    "Triste": "sad",
    "Arrabbiato": "angry",
    "Impaurito": "fearful",
    "Disgustato": "disgust",
    "Sorpreso": "surprised",
}

# Per queste emozioni proponiamo la scelta tra playlist coerente e benessere.
EMOTIONS_WITH_MODE_QUESTION = {"angry", "fearful", "sad", "disgust"}

# Destinazioni definite per la modalità benessere.
WELLBEING_DESTINATIONS = {
    "angry": "calm",
    "fearful": "calm",
    "disgust": "calm",
    "sad": "happy",
}

# Centroidi valenza-arousal RAVDESS comunicati dal professore.
# Ogni coppia è (valence, arousal), nella scala 1-9.
VA_CENTERS = {
    "happy": (8.47, 6.05),
    "calm": (6.89, 1.67),
    "sad": (2.10, 3.49),
    "angry": (2.53, 6.20),
    "fearful": (2.66, 5.45),
    "disgust": (3.32, 5.00),
    "surprised": (6.57, 5.95),
    "neutral": (5.50, 3.45),
}


# ============================================================
# INIZIALIZZAZIONE DELLO STATO DELLA SESSIONE
# ============================================================

SESSION_DEFAULTS = {
    "analisi_completata": False,
    "playlist_generata": False,
    "final_emotion": None,
    "playlist_df": None,
    "playlist_fallback_used": False,
    "playlist_mode": None,
    "playlist_destination": None,
    "questionario_aperto": False,
    "risultato_salvato": False,
    "preferenze_iniziali": None,
    "emozione_dichiarata_iniziale": None,
}

for state_key, default_value in SESSION_DEFAULTS.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


# ============================================================
# COLLEGAMENTO A GITHUB PER I RISULTATI DEL QUESTIONARIO
# ============================================================

GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_OWNER = "ilariabroggi"
GITHUB_REPO = "questionarioplaylist"
GITHUB_BRANCH = "main"
GITHUB_RESULTS_FILE = "risultati_questionario.csv"

GITHUB_API_URL = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/"
    f"{GITHUB_REPO}/contents/{GITHUB_RESULTS_FILE}"
)

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
    "requisiti_soddisfatti",
    "coerenza_playlist",
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
    "gusti_musicali",
    "canzone_preferita",
    "canzone_meno_preferita",
    "canzoni_gia_conosciute",
    "commento_valutazione",
    "frequenza_ascolto_musica",
    "scoperta_musica",
    "playlist_mood",
    "scelta_algoritmo",
    "preferenza_musica_per_mood",
    "genere",
    "eta",
    "regione_residenza",
    "istruzione",
]

def load_results_from_github():
    """Legge il CSV dei risultati e restituisce anche il suo SHA GitHub."""
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "Il segreto GITHUB_TOKEN non è configurato in Streamlit."
        )

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
    }

    response = requests.get(
        GITHUB_API_URL,
        headers=headers,
        params={"ref": GITHUB_BRANCH},
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Errore durante la lettura di risultati_questionario.csv da GitHub. "
            f"Codice HTTP: {response.status_code}. Risposta: {response.text}"
        )

    data = response.json()
    decoded_content = base64.b64decode(data["content"]).decode("utf-8")
    results_df = pd.read_csv(StringIO(decoded_content))
    return results_df, data["sha"]


def generate_participant_id(results_df):
    if results_df.empty or "id_partecipante" not in results_df.columns:
        return "0001"

    existing_ids = results_df["id_partecipante"].astype(str).str.extract(
        r"(\d+)", expand=False
    )
    numeric_ids = pd.to_numeric(existing_ids, errors="coerce").dropna()
    next_id = 1 if numeric_ids.empty else int(numeric_ids.max()) + 1
    return f"{next_id:04d}"


def save_result_to_github(new_row):
    """Aggiunge una riga al CSV su GitHub, ritentando in caso di conflitto."""
    for attempt in range(3):
        results_df, file_sha = load_results_from_github()
        new_row_df = pd.DataFrame([new_row])

        # Migra alcuni campi della vecchia versione prima di allineare lo schema.
        # In questo modo i commenti e la modalità già raccolti non vengono persi.
        if "coerenza_playlist" not in results_df.columns and "modalita_playlist" in results_df.columns:
            results_df["coerenza_playlist"] = results_df["modalita_playlist"]
        if "commento_valutazione" not in results_df.columns and "commento" in results_df.columns:
            results_df["commento_valutazione"] = results_df["commento"]

        # Allinea anche il CSV già presente su GitHub al nuovo schema:
        # aggiunge le nuove colonne vuote per i vecchi partecipanti e rimuove
        # le vecchie colonne non più utilizzate.
        for column in RESULTS_COLUMNS:
            if column not in results_df.columns:
                results_df[column] = pd.NA
            if column not in new_row_df.columns:
                new_row_df[column] = pd.NA

        results_df = results_df[RESULTS_COLUMNS]
        new_row_df = new_row_df[RESULTS_COLUMNS]
        updated_df = pd.concat([results_df, new_row_df], ignore_index=True)
        csv_content = updated_df.to_csv(index=False)
        encoded_content = base64.b64encode(
            csv_content.encode("utf-8")
        ).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        }
        payload = {
            "message": f"Salvataggio questionario {new_row['id_partecipante']}",
            "content": encoded_content,
            "sha": file_sha,
            "branch": GITHUB_BRANCH,
        }

        response = requests.put(
            GITHUB_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )

        if response.status_code in (200, 201):
            return True
        if response.status_code == 409 and attempt < 2:
            continue
        if response.status_code == 409:
            raise RuntimeError("CONFLICT_GITHUB")

        raise RuntimeError(
            "Errore durante il salvataggio del risultato su GitHub. "
            f"Codice HTTP: {response.status_code}. Risposta: {response.text}"
        )

    return False


# ============================================================
# CARICAMENTO DEL MODELLO SER VOCALE
# ============================================================

try:
    centroids = pd.read_csv(BASE_DIR / "centroidi_emozioni.csv", index_col=0)
    scaler = joblib.load(BASE_DIR / "scaler_mfcc.pkl")
    with open(BASE_DIR / "feature_cols.json", "r", encoding="utf-8") as file:
        feature_cols = json.load(file)
except Exception as exc:
    st.error(
        "Non è stato possibile caricare i file del modello vocale. "
        "Controlla che centroidi_emozioni.csv, scaler_mfcc.pkl e "
        f"feature_cols.json siano presenti nel repository. Dettaglio: {exc}"
    )
    st.stop()

centroids.index = [str(emotion).strip().lower() for emotion in centroids.index]
centroids = centroids[~centroids.index.duplicated(keep="first")]

missing_ser_emotions = [emotion for emotion in EMOTIONS if emotion not in centroids.index]
if missing_ser_emotions:
    st.error(
        "Il file centroidi_emozioni.csv non contiene tutti gli otto centroidi "
        "emotivi necessari per questa versione dell'app: "
        f"{', '.join(missing_ser_emotions)}. "
        "Non è possibile distinguere tutte le otto emozioni se i centroidi "
        "del modello vocale non sono disponibili."
    )
    st.stop()

missing_centroid_features = [feature for feature in feature_cols if feature not in centroids.columns]
if missing_centroid_features:
    st.error(
        "Nel file centroidi_emozioni.csv mancano alcune feature MFCC: "
        f"{', '.join(missing_centroid_features)}"
    )
    st.stop()


# ============================================================
# CONVERSIONE DELLE RISPOSTE DEL QUESTIONARIO
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
    "Colonne sonore e musica classica": "Classical",
}

language_mapping = {
    "Italiano": "italian",
    "Inglese": "english",
    "Spagnolo": "spanish",
    "Portoghese": "portuguese",
    "Francese": "french",
    "Tedesco": "german",
}

POPULARITY_POPULAR = "Canzoni popolari"
POPULARITY_NICHE = "Canzoni poco conosciute"


# ============================================================
# ESTRAZIONE DELLE FEATURE MFCC E PREVISIONE SER
# ============================================================

def extract_mfcc_features(audio_file):
    audio_bytes = audio_file.getvalue()
    signal, sample_rate = librosa.load(BytesIO(audio_bytes), sr=48000)

    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    feature_vector = np.concatenate([mfcc_mean, mfcc_std])

    if len(feature_vector) != len(feature_cols):
        raise ValueError(
            "Il numero di feature MFCC estratte non coincide con quello "
            "atteso dal modello. Controlla feature_cols.json."
        )

    return pd.DataFrame([feature_vector], columns=feature_cols)


def predict_emotion_from_two_recordings(audio_1, audio_2):
    """Classifica il punto medio delle due registrazioni rispetto agli 8 centroidi."""
    features_1 = extract_mfcc_features(audio_1)
    features_2 = extract_mfcc_features(audio_2)

    standardized_1 = scaler.transform(features_1[feature_cols])
    standardized_2 = scaler.transform(features_2[feature_cols])
    mean_point = (standardized_1[0] + standardized_2[0]) / 2

    distances = {}
    for emotion in EMOTIONS:
        centroid = centroids.loc[emotion, feature_cols].to_numpy(dtype=float)
        distances[emotion] = float(np.linalg.norm(mean_point - centroid))

    if not distances:
        raise ValueError("Non è stato possibile calcolare le distanze dai centroidi vocali.")

    return min(distances, key=distances.get), distances


# ============================================================
# CARICAMENTO DEGLI OTTO DATASET MUSICALI
# ============================================================

REQUIRED_MUSIC_COLUMNS = [
    "id",
    "name",
    "album_name",
    "artists",
    "year",
    "genre",
    "popularity",
    "language",
    "valence_va",
    "arousal_va",
    "emotion_va",
    "emotion_va_distance",
]
OPTIONAL_MUSIC_COLUMNS = []  # Le colonne non usate dalla playlist non vengono caricate, per contenere la memoria.


def normalize_column_name(value):
    value = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


MUSIC_DATASET_FILES = {
    "happy": "songs_clean_happy.csv",
    "surprised": "songs_clean_surprised.csv",
    "neutral": "songs_clean_neutral.csv",
    "calm": "songs_clean_calm.csv",
    "angry": "songs_clean_angry.csv",
    "fearful": "songs_clean_fearful.csv",
    "disgust": "songs_clean_disgust.csv",
    "sad": "songs_clean_sad.csv",
}


def find_emotion_dataset_file(emotion):
    """Restituisce il percorso del CSV previsto per una specifica emotion_va."""
    emotion = str(emotion).strip().lower()

    if emotion not in MUSIC_DATASET_FILES:
        raise ValueError(f"Emozione musicale non riconosciuta: {emotion}")

    file_name = MUSIC_DATASET_FILES[emotion]
    file_path = BASE_DIR / file_name

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Non è stato trovato il dataset dell'emozione '{emotion}'. "
            f"Il file atteso è '{file_name}' nella cartella {BASE_DIR}. "
            "Controlla che il nome sia esattamente questo e che il file sia nel repository."
        )

    return file_path

@st.cache_data(show_spinner=False)
def load_music_dataset(emotion):
    """Carica un dataset musicale singolo in base a emotion_va."""
    emotion = str(emotion).strip().lower()
    if emotion not in EMOTIONS:
        raise ValueError(f"Emozione musicale non riconosciuta: {emotion}")

    file_path = find_emotion_dataset_file(emotion)
    header = pd.read_csv(file_path, nrows=0)
    column_map = {normalize_column_name(column): column for column in header.columns}

    missing_columns = [
        column for column in REQUIRED_MUSIC_COLUMNS
        if column not in column_map
    ]
    if missing_columns:
        raise ValueError(
            f"Nel dataset {file_path.name} mancano le colonne: {', '.join(missing_columns)}. "
            f"Colonne trovate: {', '.join(map(str, header.columns))}"
        )

    requested_columns = [column_map[column] for column in REQUIRED_MUSIC_COLUMNS]
    requested_columns += [
        column_map[column] for column in OPTIONAL_MUSIC_COLUMNS if column in column_map
    ]
    df = pd.read_csv(
        file_path,
        usecols=list(dict.fromkeys(requested_columns)),
        low_memory=False,
    )
    df.columns = [normalize_column_name(column) for column in df.columns]

    df["id"] = df["id"].astype("string").str.strip()
    df["emotion_va"] = df["emotion_va"].astype("string").str.strip().str.lower()
    df["language"] = df["language"].astype("string").str.strip().str.lower()
    df["genre"] = df["genre"].astype("string").str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["valence_va"] = pd.to_numeric(df["valence_va"], errors="coerce")
    df["arousal_va"] = pd.to_numeric(df["arousal_va"], errors="coerce")
    df["emotion_va_distance"] = pd.to_numeric(df["emotion_va_distance"], errors="coerce")

    df = df.dropna(
        subset=[
            "id", "name", "album_name", "artists", "year", "genre",
            "popularity", "language", "valence_va", "arousal_va",
            "emotion_va", "emotion_va_distance",
        ]
    ).copy()
    df = df[df["id"].ne("")].copy()

    # Il file di partizione dovrebbe contenere soltanto l'emozione richiesta;
    # il filtro rende il comportamento sicuro anche se un file è stato creato male.
    df = df[df["emotion_va"].eq(emotion)].copy()

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_all_music_datasets():
    """Unisce gli otto dataset per poter cercare brani sui punti intermedi."""
    frames = [load_music_dataset(emotion) for emotion in EMOTIONS]
    all_songs = pd.concat(frames, ignore_index=True)
    all_songs = all_songs.drop_duplicates(subset="id", keep="first")
    return all_songs.reset_index(drop=True)


# ============================================================
# FILTRI E ORDINAMENTO PER LA PLAYLIST COERENTE
# ============================================================

def filter_by_preferences(df, preferences, tier):
    """Applica i filtri; il genere resta sempre obbligatorio.

    tier 0: genere + periodo + lingua
    tier 1: genere + periodo
    tier 2: solo genere
    """
    music_genre = genre_mapping[preferences["genre"]]
    filtered = df[df["genre"].eq(music_genre)].copy()

    if tier <= 1:
        filtered = filtered[
            filtered["year"].ge(preferences["start_year"])
            & filtered["year"].le(preferences["end_year"])
        ].copy()

    if tier == 0:
        music_language = language_mapping[preferences["language"]]
        filtered = filtered[filtered["language"].eq(music_language)].copy()

    return filtered


def rank_coherent_block(df, popularity_preference):
    if df.empty:
        return df.copy()
    if popularity_preference == POPULARITY_POPULAR:
        return df.sort_values("popularity", ascending=False, kind="stable").copy()
    return df.sort_values("emotion_va_distance", ascending=True, kind="stable").copy()


def generate_coherent_playlist(emotion, preferences, target_songs=7):
    """Playlist coerente: tutti i brani devono avere l'emozione stimata."""
    emotion_df = load_music_dataset(emotion)
    last_candidate_set = emotion_df.iloc[0:0].copy()

    for tier in (0, 1, 2):
        candidates = filter_by_preferences(emotion_df, preferences, tier)
        last_candidate_set = candidates
        ranked = rank_coherent_block(candidates, preferences["popularity_preference"])

        if len(ranked) >= target_songs:
            return ranked.head(target_songs).reset_index(drop=True), tier > 0

    ranked = rank_coherent_block(last_candidate_set, preferences["popularity_preference"])
    return ranked.head(target_songs).reset_index(drop=True), True


# ============================================================
# GEOMETRIA DEL PERCORSO VALENZA-AROUSAL
# ============================================================

def coordinates_for_emotion(emotion):
    if emotion not in VA_CENTERS:
        raise ValueError(f"Non è disponibile il centro valenza-arousal per '{emotion}'.")
    return np.asarray(VA_CENTERS[emotion], dtype=float)


def intermediate_points(start_emotion, destination_emotion):
    """Restituisce i tre punti al 25%, 50% e 75% della traiettoria."""
    start = coordinates_for_emotion(start_emotion)
    destination = coordinates_for_emotion(destination_emotion)
    return [start + t * (destination - start) for t in (0.25, 0.50, 0.75)]


def select_endpoint_songs(candidates, emotion, count, popularity_preference, excluded_ids):
    pool = candidates[candidates["emotion_va"].eq(emotion)].copy()
    pool = pool[~pool["id"].astype(str).isin(excluded_ids)].copy()

    if popularity_preference == POPULARITY_POPULAR:
        pool = pool.sort_values("popularity", ascending=False, kind="stable")
    else:
        pool = pool.sort_values("emotion_va_distance", ascending=True, kind="stable")

    selected = pool.head(count).copy()
    return selected.reset_index(drop=True)


def select_middle_songs(candidates, start_emotion, destination_emotion,
                        popularity_preference, excluded_ids, allow_partial=False):
    """Seleziona tre brani intermedi e li ordina dall'inizio alla fine.

    Popolari: fra i brani la cui proiezione cade dentro il segmento, prende i
    tre più popolari e poi li ordina lungo la traiettoria.

    Poco conosciute/rappresentative: sceglie il brano più vicino a ciascuno dei
    tre punti intermedi (25%, 50%, 75%), senza riutilizzare ID.
    """
    start = coordinates_for_emotion(start_emotion)
    destination = coordinates_for_emotion(destination_emotion)
    direction = destination - start
    length_squared = float(np.dot(direction, direction))

    pool = candidates[~candidates["id"].astype(str).isin(excluded_ids)].copy()
    if pool.empty:
        return pool

    xy = pool[["valence_va", "arousal_va"]].to_numpy(dtype=float)
    projection_t = ((xy - start) @ direction) / length_squared
    pool = pool.assign(_trajectory_t=projection_t)

    if popularity_preference == POPULARITY_POPULAR:
        # Consideriamo i brani la cui proiezione cade fra i due centri.
        # Non imponiamo una soglia di distanza laterale non prevista dal metodo.
        middle_pool = pool[pool["_trajectory_t"].gt(0) & pool["_trajectory_t"].lt(1)].copy()
        middle_pool = middle_pool.sort_values("popularity", ascending=False, kind="stable")
        if len(middle_pool) < 3 and not allow_partial:
            return middle_pool.iloc[0:0].drop(columns=["_trajectory_t"], errors="ignore")
        selected = middle_pool.head(3).copy()
        selected = selected.sort_values("_trajectory_t", ascending=True, kind="stable")
        return selected.drop(columns=["_trajectory_t"], errors="ignore").reset_index(drop=True)

    # Preferenza per brani più rappresentativi del punto intermedio.
    targets = intermediate_points(start_emotion, destination_emotion)
    selected_rows = []
    used_ids = set(str(value) for value in excluded_ids)

    for target in targets:
        remaining = pool[~pool["id"].astype(str).isin(used_ids)].copy()
        if remaining.empty:
            if allow_partial:
                continue
            return candidates.iloc[0:0].copy()

        remaining["_target_distance"] = np.sqrt(
            (remaining["valence_va"] - target[0]) ** 2
            + (remaining["arousal_va"] - target[1]) ** 2
        )
        best = remaining.sort_values("_target_distance", ascending=True, kind="stable").head(1)
        if best.empty:
            if allow_partial:
                continue
            return candidates.iloc[0:0].copy()

        selected_rows.append(best.drop(columns=["_target_distance", "_trajectory_t"]))
        used_ids.add(str(best.iloc[0]["id"]))

    if len(selected_rows) < 3 and not allow_partial:
        return candidates.iloc[0:0].copy()
    if not selected_rows:
        return candidates.iloc[0:0].copy()

    return pd.concat(selected_rows, ignore_index=True)


def build_wellbeing_playlist(candidates, start_emotion, destination_emotion,
                             popularity_preference, allow_partial=False):
    """Crea la sequenza 2 partenza + 3 intermedi + 2 arrivo."""
    if candidates.empty:
        return candidates.copy() if allow_partial else None

    start_songs = select_endpoint_songs(
        candidates, start_emotion, 2, popularity_preference, excluded_ids=set()
    )
    end_songs = select_endpoint_songs(
        candidates, destination_emotion, 2, popularity_preference,
        excluded_ids=set(start_songs["id"].astype(str))
    )

    if not allow_partial and (len(start_songs) < 2 or len(end_songs) < 2):
        return None

    endpoint_ids = set(start_songs["id"].astype(str)) | set(end_songs["id"].astype(str))
    middle_songs = select_middle_songs(
        candidates=candidates,
        start_emotion=start_emotion,
        destination_emotion=destination_emotion,
        popularity_preference=popularity_preference,
        excluded_ids=endpoint_ids,
        allow_partial=allow_partial,
    )

    if not allow_partial and len(middle_songs) < 3:
        return None

    playlist = pd.concat(
        [start_songs, middle_songs, end_songs],
        ignore_index=True,
    )
    playlist = playlist.drop_duplicates(subset="id", keep="first")

    if not allow_partial and len(playlist) != 7:
        return None

    return playlist.reset_index(drop=True)


def generate_wellbeing_playlist(start_emotion, preferences, target_songs=7):
    """Costruisce il percorso emotivo e rilassa lingua, poi periodo.

    Il genere rimane obbligatorio. Gli estremi sono vincolati alle emozioni
    di partenza e destinazione; le canzoni intermedie vengono scelte nel piano VA.
    """
    destination = WELLBEING_DESTINATIONS[start_emotion]
    all_songs = load_all_music_datasets()
    last_candidates = all_songs.iloc[0:0].copy()

    for tier in (0, 1, 2):
        candidates = filter_by_preferences(all_songs, preferences, tier)
        last_candidates = candidates
        playlist = build_wellbeing_playlist(
            candidates=candidates,
            start_emotion=start_emotion,
            destination_emotion=destination,
            popularity_preference=preferences["popularity_preference"],
            allow_partial=(tier == 2),
        )

        if playlist is not None and (len(playlist) == target_songs or tier == 2):
            return playlist.head(target_songs).reset_index(drop=True), tier > 0 or len(playlist) < target_songs, destination

    # Normalmente il codice torna dal livello 2. Questo è un fallback prudenziale.
    partial = build_wellbeing_playlist(
        candidates=last_candidates,
        start_emotion=start_emotion,
        destination_emotion=destination,
        popularity_preference=preferences["popularity_preference"],
        allow_partial=True,
    )
    if partial is None:
        partial = all_songs.iloc[0:0].copy()
    return partial.head(target_songs).reset_index(drop=True), True, destination


def generate_playlist_for_mode(emotion, mode, preferences):
    if mode == "coerente":
        playlist, fallback_used = generate_coherent_playlist(
            emotion=emotion,
            preferences=preferences,
            target_songs=7,
        )
        return playlist, fallback_used, None

    if mode == "benessere":
        playlist, fallback_used, destination = generate_wellbeing_playlist(
            start_emotion=emotion,
            preferences=preferences,
            target_songs=7,
        )
        return playlist, fallback_used, destination

    raise ValueError(f"Modalità playlist non riconosciuta: {mode}")


# ============================================================
# LINK SPOTIFY E YOUTUBE
# ============================================================

def create_spotify_link(song_id):
    return f"https://open.spotify.com/track/{str(song_id).strip()}"


def create_youtube_link(title, artist):
    query = quote_plus(f"{artist} {title}")
    return f"https://www.youtube.com/results?search_query={query}"


# ============================================================
# TITOLO E INTRODUZIONE
# ============================================================

st.title("🎵 Emozioni, voce e musica")
st.subheader(
    "Un progetto di tesi sulla relazione tra emozioni vocali e preferenze musicali"
)

st.write(
    """Gentile partecipante,

Ti ringraziamo per aver scelto di partecipare a questo questionario sviluppato
da una laureanda dell'Università degli Studi di Modena e Reggio Emilia.

Prima di procedere con la compilazione, ti preghiamo di leggere attentamente le
seguenti informazioni riguardanti la privacy e l'utilizzo delle tue risposte.

Tutte le informazioni raccolte in questa ricerca saranno trattate nel rispetto
delle vigenti leggi D.Lgs. 196/2003 sulla privacy e UE GDPR 679/2016 sulla
protezione dei dati personali. Ti chiediamo di non includere informazioni
personali sensibili o identificabili nelle risposte al questionario, come nomi
completi, indirizzi, numeri di telefono, indirizzi email, numeri di documenti
d'identità o altre informazioni che potrebbero identificarti direttamente.

I dati, raccolti ed elaborati in forma aggregata e anonima, potranno essere
inseriti in pubblicazioni e/o presentati a congressi, seminari o lezioni.
Proseguendo con la compilazione del questionario, confermi di aver letto e
compreso le avvertenze sulla privacy, di essere maggiorenne, di aderire
volontariamente alla realizzazione della ricerca in qualità di partecipante,
di essere a conoscenza che i dati ricavati, nell'assoluto anonimato, saranno
trattati esclusivamente per fini didattici e di ricerca e di essere consapevole
che è prevista la possibilità di ottenere la restituzione dei dati raccolti una
volta inviati.

Per eventuali chiarimenti è possibile contattare:

Ilaria Broggi (302189@studenti.unimore.it)

Grazie mille per il tuo prezioso contributo!"""
)
privacy_consent = st.checkbox(
    "Ho letto e compreso l'informativa, dichiaro di essere maggiorenne e scelgo "
    "volontariamente di partecipare alla ricerca.",
    key="privacy_consent",
)
st.divider()


# ============================================================
# QUESTIONARIO PRE-REGISTRAZIONE
# Tutte le risposte obbligatorie partono senza selezione.
# ============================================================

st.header("🎧 Le tue preferenze musicali")
st.write("Prima di registrare la voce, rispondi a tutte le domande seguenti.")

GENRE_OPTIONS = [
    "Rock", "Pop", "Elettronica", "Folk", "Country", "Hip-Hop",
    "Rhythm and Blues", "Jazz", "Blues", "Colonne sonore e musica classica",
]
LANGUAGE_OPTIONS = ["Italiano", "Inglese", "Spagnolo", "Portoghese", "Francese", "Tedesco"]
EMOTION_OPTIONS_IT = [
    "Neutro", "Calmo", "Felice", "Triste", "Arrabbiato",
    "Impaurito", "Disgustato", "Sorpreso",
]
YEARS = list(range(1900, 2026))

# Non blocchiamo i campi dopo l'analisi: le preferenze usate nella playlist
# vengono comunque salvate in sessione al momento dell'invio dell'analisi.
genere = st.selectbox(
    "Qual è il tuo genere musicale preferito?",
    GENRE_OPTIONS,
    index=None,
    placeholder="Seleziona un genere",
    key="pref_genere",
)

lingua = st.selectbox(
    "In quale lingua preferisci ascoltare musica?",
    LANGUAGE_OPTIONS,
    index=None,
    placeholder="Seleziona una lingua",
    key="pref_lingua",
)

anno_inizio = st.selectbox(
    "Da quale anno vuoi ascoltare musica?",
    YEARS,
    index=None,
    placeholder="Seleziona l'anno iniziale",
    key="pref_anno_inizio",
)

anno_fine = st.selectbox(
    "Fino a quale anno vuoi ascoltare musica?",
    YEARS,
    index=None,
    placeholder="Seleziona l'anno finale",
    key="pref_anno_fine",
)

if anno_inizio is not None and anno_fine is not None and anno_fine < anno_inizio:
    st.warning("L'anno finale deve essere uguale o successivo all'anno iniziale.")

popularity_preference = st.radio(
    "Preferisci ascoltare canzoni popolari o canzoni poco conosciute?",
    [POPULARITY_POPULAR, POPULARITY_NICHE],
    index=None,
    key="pref_popolarita",
)

st.subheader("💭 Come ti senti in questo momento?")
st.write(
    "Scegli l'emozione o lo stato d'animo che ritieni maggiormente rappresentativo "
    "di come ti senti in questo momento. La risposta sarà confrontata con quella "
    "stimata successivamente attraverso l'analisi della voce."
)

emozione_dichiarata = st.radio(
    "Come descriveresti il tuo stato emotivo attuale?",
    EMOTION_OPTIONS_IT,
    index=None,
    key="pref_emozione_dichiarata",
)

st.divider()


# ============================================================
# REGISTRAZIONE DELLA VOCE
# ============================================================

st.header("🎙️ Registrazione della voce")
st.write(
    "Pronuncia ciascuna frase cercando di far trasparire attraverso la voce il "
    "tuo stato emotivo attuale. Non cercare di interpretare artificialmente "
    "un'emozione: parla nel modo più naturale possibile."
)

st.subheader("Registrazione 1")
st.write("Pronuncia la seguente frase:")
st.info("🗣️ «I bambini stanno parlando vicino alla porta.»")
audio_1 = st.audio_input("Registra la prima frase", sample_rate=48000, key="audio_1")

st.subheader("Registrazione 2")
st.write("Pronuncia la seguente frase:")
st.info("🗣️ «I cani sono seduti vicino alla porta.»")
audio_2 = st.audio_input("Registra la seconda frase", sample_rate=48000, key="audio_2")

st.divider()


# ============================================================
# RIEPILOGO DELLE RISPOSTE PRE-REGISTRAZIONE
# ============================================================

st.header("📋 Riepilogo")
st.write("Prima di procedere, verifica le informazioni inserite.")
st.write(f"**Genere musicale preferito:** {genere if genere is not None else 'Non selezionato'}")
st.write(f"**Lingua preferita:** {lingua if lingua is not None else 'Non selezionata'}")
st.write(
    "**Periodo musicale preferito:** "
    f"{anno_inizio if anno_inizio is not None else '—'}–"
    f"{anno_fine if anno_fine is not None else '—'}"
)
st.write(
    "**Preferenza:** "
    f"{popularity_preference if popularity_preference is not None else 'Non selezionata'}"
)
st.write(
    "**Emozione dichiarata:** "
    f"{emozione_dichiarata if emozione_dichiarata is not None else 'Non selezionata'}"
)

st.divider()
procedi = st.button("Analizza la mia voce 🎵", type="primary", use_container_width=True)


# ============================================================
# AVVIO DELL'ANALISI VOCALE
# ============================================================

if procedi:
    missing_pre_answers = []
    if genere is None:
        missing_pre_answers.append("genere musicale")
    if lingua is None:
        missing_pre_answers.append("lingua")
    if anno_inizio is None:
        missing_pre_answers.append("anno iniziale")
    if anno_fine is None:
        missing_pre_answers.append("anno finale")
    if popularity_preference is None:
        missing_pre_answers.append("preferenza di popolarità")
    if emozione_dichiarata is None:
        missing_pre_answers.append("emozione dichiarata")

    if not privacy_consent:
        missing_pre_answers.append("conferma di aver letto l'informativa privacy")

    if missing_pre_answers:
        st.error("Rispondi a tutte le domande e conferma l'informativa prima di procedere.")
    elif anno_fine < anno_inizio:
        st.error("Seleziona un periodo valido: l'anno finale deve essere uguale o successivo all'anno iniziale.")
    elif audio_1 is None or audio_2 is None:
        st.error("Registra entrambe le frasi prima di avviare l'analisi.")
    else:
        try:
            with st.spinner("Analisi della voce in corso..."):
                final_emotion, _ = predict_emotion_from_two_recordings(audio_1, audio_2)

            st.session_state.final_emotion = final_emotion
            st.session_state.preferenze_iniziali = {
                "genre": genere,
                "language": lingua,
                "start_year": int(anno_inizio),
                "end_year": int(anno_fine),
                "popularity_preference": popularity_preference,
            }
            st.session_state.emozione_dichiarata_iniziale = emozione_dichiarata
            st.session_state.analisi_completata = True
            st.session_state.playlist_generata = False
            st.session_state.playlist_df = None
            st.session_state.playlist_fallback_used = False
            st.session_state.playlist_mode = None
            st.session_state.playlist_destination = None
            st.session_state.questionario_aperto = False
            st.session_state.risultato_salvato = False

            # Per happy, surprised, neutral e calm partiamo direttamente
            # con la playlist coerente. Le altre emozioni richiedono una scelta.
            if final_emotion not in EMOTIONS_WITH_MODE_QUESTION:
                with st.spinner("Sto cercando le canzoni più adatte a te..."):
                    playlist, fallback_used, destination = generate_playlist_for_mode(
                        emotion=final_emotion,
                        mode="coerente",
                        preferences=st.session_state.preferenze_iniziali,
                    )
                st.session_state.playlist_df = playlist
                st.session_state.playlist_fallback_used = fallback_used
                st.session_state.playlist_mode = "coerente"
                st.session_state.playlist_destination = destination
                st.session_state.playlist_generata = True

        except Exception as exc:
            st.error("Si è verificato un errore durante l'analisi o la ricerca della playlist.")
            st.exception(exc)


# ============================================================
# RISULTATO DELL'EMOZIONE E SCELTA DELLA MODALITÀ
# ============================================================

if st.session_state.analisi_completata:
    final_emotion = st.session_state.final_emotion
    declared_emotion_text = st.session_state.emozione_dichiarata_iniziale
    declared_emotion = DECLARED_EMOTION_MAPPING[declared_emotion_text]

    st.divider()
    st.success("Analisi completata! 🎉")
    st.header("🧠 Emozione stimata")
    st.subheader(EMOTION_ITALIAN.get(final_emotion, final_emotion.capitalize()))

    if final_emotion != declared_emotion:
        st.info(
            f"Hai indicato **{declared_emotion_text}**, mentre l'analisi della voce "
            f"ha stimato **{EMOTION_ITALIAN[final_emotion]}**."
        )
    else:
        st.info(
            f"Hai indicato **{declared_emotion_text}** e l'analisi della voce "
            f"ha stimato **{EMOTION_ITALIAN[final_emotion]}**."
        )

    if final_emotion in EMOTIONS_WITH_MODE_QUESTION and not st.session_state.playlist_generata:
        st.divider()
        st.header("🎵 Che tipo di playlist preferisci?")
        st.write(
            "Puoi ascoltare una playlist coerente con il tuo stato d'animo attuale "
            "oppure una playlist pensata per accompagnare gradualmente un percorso "
            "emotivo verso un altro stato d'animo. Quest'ultima è una proposta musicale, "
            "non una promessa di miglioramento del benessere psicologico."
        )

        mode_choice = st.radio(
            "Preferisci una playlist:",
            [
                "Coerente con il mio stato d'animo attuale",
                "Per accompagnare il mio benessere emotivo",
            ],
            index=None,
            key="scelta_modalita_playlist",
        )

        genera_playlist = st.button(
            "Genera la playlist 🎶",
            type="primary",
            use_container_width=True,
            key="genera_playlist_modalita",
        )

        if genera_playlist:
            if mode_choice is None:
                st.error("Seleziona una modalità prima di generare la playlist.")
            else:
                mode = "coerente" if mode_choice.startswith("Coerente") else "benessere"
                try:
                    with st.spinner("Sto costruendo la tua playlist..."):
                        playlist, fallback_used, destination = generate_playlist_for_mode(
                            emotion=final_emotion,
                            mode=mode,
                            preferences=st.session_state.preferenze_iniziali,
                        )
                    st.session_state.playlist_df = playlist
                    st.session_state.playlist_fallback_used = fallback_used
                    st.session_state.playlist_mode = mode
                    st.session_state.playlist_destination = destination
                    st.session_state.playlist_generata = True
                    st.rerun()
                except Exception as exc:
                    st.error("Si è verificato un errore durante la generazione della playlist.")
                    st.exception(exc)

    # ========================================================
    # VISUALIZZAZIONE DELLA PLAYLIST
    # ========================================================

    if st.session_state.playlist_generata:
        playlist_df = st.session_state.playlist_df
        mode = st.session_state.playlist_mode
        destination = st.session_state.playlist_destination

        st.divider()
        st.header("🎵 La tua playlist")

        if mode == "benessere" and destination:
            st.write(
                "Questa playlist è stata organizzata lungo un percorso tra "
                f"**{EMOTION_ITALIAN[final_emotion].lower()}** e "
                f"**{EMOTION_ITALIAN[destination].lower()}**, cercando di passare "
                "gradualmente da un estremo all'altro."
            )
        else:
            st.write(
                "Questa playlist raccoglie canzoni associate all'emozione "
                f"**{EMOTION_ITALIAN[final_emotion].lower()}**."
            )

        if st.session_state.playlist_fallback_used:
            st.info(
                "Non tutte le canzoni rispettavano contemporaneamente tutte le "
                "preferenze. Per cercare brani sufficienti, il sistema ha prima "
                "allentato il vincolo sulla lingua e, se necessario, quello sul periodo. "
                "Il genere e il percorso/emozione previsti sono stati mantenuti."
            )

        if playlist_df is None or playlist_df.empty:
            st.warning(
                "Non sono state trovate canzoni sufficienti con i vincoli selezionati. "
                "Prova con un altro genere o un intervallo temporale più ampio."
            )
        else:
            st.write(f"Sono state selezionate **{len(playlist_df)} canzoni**.")
            if len(playlist_df) < 7:
                st.warning(
                    "Nel dataset non erano disponibili sette canzoni distinte che "
                    "rispettassero i vincoli mantenuti. La playlist contiene quindi "
                    f"{len(playlist_df)} brani."
                )

            for position, (_, song) in enumerate(playlist_df.iterrows(), start=1):
                spotify_url = create_spotify_link(song["id"])
                youtube_url = create_youtube_link(song["name"], song["artists"])
                st.subheader(f"{position}. {song['name']}")
                st.write(f"**Artista:** {song['artists']}")
                st.write(f"**Album:** {song['album_name']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.link_button("Spotify", spotify_url, use_container_width=True)
                with col2:
                    st.link_button("YouTube", youtube_url, use_container_width=True)

            # ====================================================
            # APERTURA DEL QUESTIONARIO FINALE
            # ====================================================
            st.divider()
            st.header("🎧 Hai finito di ascoltare la playlist?")
            st.write(
                "Quando hai finito di ascoltare le canzoni, clicca sul pulsante "
                "qui sotto per rispondere a poche domande sulla playlist."
            )
            if st.button("Ho finito di ascoltare 🎶", use_container_width=True, key="apri_questionario_finale"):
                st.session_state.questionario_aperto = True
                st.rerun()


# ============================================================
# QUESTIONARIO FINALE
# Tutte le risposte sono obbligatorie tranne il commento facoltativo.
# ============================================================

if (
    st.session_state.analisi_completata
    and st.session_state.playlist_generata
    and st.session_state.questionario_aperto
    and not st.session_state.risultato_salvato
    and st.session_state.playlist_df is not None
    and not st.session_state.playlist_df.empty
):
    playlist_df = st.session_state.playlist_df
    final_emotion = st.session_state.final_emotion
    mode = st.session_state.playlist_mode
    destination = st.session_state.playlist_destination

    st.divider()
    st.header("📝 Questionario finale")
    st.write(
        "Rispondi alle seguenti domande pensando alla playlist che hai appena ascoltato. "
        "Il questionario è diviso in tre sezioni."
    )

    song_labels = []
    for position, (_, song) in enumerate(playlist_df.iterrows(), start=1):
        song_labels.append(
            f"{position} — {str(song['name']).strip()} — {str(song['artists']).strip()}"
        )

    # --------------------------------------------------------
    # SEZIONE 1: VALUTAZIONE DELLA PLAYLIST
    # --------------------------------------------------------
    st.header("1. Valutazione della playlist")

    scale_labels = [
        "1 — Per nulla",
        "2 — Poco",
        "3 — Moderatamente",
        "4 — Molto",
        "5 — Moltissimo",
    ]
    scale_values = {label: int(label[0]) for label in scale_labels}

    st.subheader("1.1 Quanto ti è piaciuta la playlist?")
    gradimento_label = st.radio(
        "1 = Per nulla; 5 = Moltissimo",
        scale_labels,
        index=None,
        key="final_gradimento_playlist",
    )
    gradimento_playlist = scale_values.get(gradimento_label)

    st.subheader(
        "1.2 Quanto ritieni che la playlist fosse coerente con il tuo stato emotivo "
        "al momento della registrazione?"
    )
    coerenza_emotiva_label = st.radio(
        "1 = Per nulla coerente; 5 = Completamente coerente",
        [
            "1 — Per nulla coerente",
            "2 — Poco coerente",
            "3 — Moderatamente coerente",
            "4 — Molto coerente",
            "5 — Completamente coerente",
        ],
        index=None,
        key="final_coerenza_emotiva",
    )
    coerenza_emotiva = (
        int(coerenza_emotiva_label[0]) if coerenza_emotiva_label else None
    )

    st.subheader("1.3 Quanto ritieni che la playlist fosse coerente con i tuoi gusti musicali?")
    gusti_musicali_label = st.radio(
        "1 = Per nulla coerente; 5 = Completamente coerente",
        [
            "1 — Per nulla coerente",
            "2 — Poco coerente",
            "3 — Moderatamente coerente",
            "4 — Molto coerente",
            "5 — Completamente coerente",
        ],
        index=None,
        key="final_gusti_musicali",
    )
    gusti_musicali = int(gusti_musicali_label[0]) if gusti_musicali_label else None

    st.subheader("1.4 Quale canzone ti è piaciuta di più?")
    canzone_preferita = st.radio(
        "Seleziona una canzone:",
        song_labels,
        index=None,
        key="final_canzone_preferita",
    )

    st.subheader("1.5 Quale canzone ti è piaciuta di meno?")
    canzone_meno_preferita = st.radio(
        "Seleziona una canzone:",
        song_labels,
        index=None,
        key="final_canzone_meno_preferita",
    )

    st.subheader("1.6 Quanti dei brani presenti nella playlist conoscevi già?")
    num_songs = len(playlist_df)
    known_options = ["Nessuno"] + [str(number) for number in range(1, num_songs + 1)]
    known_options_map = {"Nessuno": 0, **{str(n): n for n in range(1, num_songs + 1)}}
    canzoni_gia_conosciute_label = st.radio(
        "Indica il numero di brani che conoscevi già:",
        known_options,
        index=None,
        key="final_canzoni_gia_conosciute",
    )
    canzoni_gia_conosciute = (
        known_options_map.get(canzoni_gia_conosciute_label)
        if canzoni_gia_conosciute_label is not None
        else None
    )

    st.subheader("1.7 Se vuoi, puoi lasciare un commento sulla playlist o spiegare cosa avresti cambiato.")
    commento_valutazione = st.text_area(
        "Commento facoltativo",
        placeholder="Scrivi qui eventuali osservazioni, commenti o suggerimenti...",
        key="final_commento_valutazione",
    )

    # --------------------------------------------------------
    # SEZIONE 2: PROFILO MUSICALE DELL'UTENTE
    # --------------------------------------------------------
    st.divider()
    st.header("2. Costruzione del profilo musicale")

    st.subheader("2.1 Quanto tempo ascolti musica mediamente durante una giornata?")
    frequenza_ascolto_musica = st.radio(
        "Seleziona una risposta:",
        [
            "Meno di 30 minuti",
            "Da 30 minuti a meno di 1 ora",
            "Da 1 ora a meno di 2 ore",
            "Da 2 ore a meno di 3 ore",
            "Più di 3 ore",
        ],
        index=None,
        key="final_frequenza_ascolto_musica",
    )

    st.subheader("2.2 Quando ascolti musica, come scegli generalmente cosa ascoltare?")
    scelta_algoritmo = st.radio(
        "1 = Scelgo personalmente; 5 = Lascio scegliere il sistema",
        [
            "1 — Scelgo personalmente",
            "2 — Principalmente scelgo io, ma a volte lascio scegliere il sistema",
            "3 — Scelgo io e lascio scegliere al sistema in egual misura",
            "4 — Un po' scelgo io, ma principalmente lascio scegliere il sistema",
            "5 — Lascio che sia il sistema a scegliere per me",
        ],
        index=None,
        key="final_scelta_algoritmo",
    )
    scelta_algoritmo = int(scelta_algoritmo[0]) if scelta_algoritmo else None

    st.subheader("2.3 Quando ascolti musica, preferisci riascoltare brani che conosci oppure scoprire musica nuova?")
    scoperta_musica_label = st.radio(
        "1 = Quasi solo brani che conosco; 5 = Quasi solo brani che non conosco",
        [
            "1 — Quasi solo brani che conosco",
            "2 — Prevalentemente brani che conosco",
            "3 — Entrambe le cose in egual misura",
            "4 — Prevalentemente musica nuova",
            "5 — Quasi solo brani che non conosco",
        ],
        index=None,
        key="final_scoperta_musica",
    )
    scoperta_musica = int(scoperta_musica_label[0]) if scoperta_musica_label else None

    st.subheader("2.4 Quanto spesso scegli musica o playlist in base al tuo stato d'animo?")
    playlist_mood = st.radio(
        "Seleziona una risposta:",
        ["Mai", "Raramente", "Qualche volta", "Spesso", "Sempre"],
        index=None,
        key="final_playlist_mood",
    )

    st.subheader("2.5 Quando scegli musica in base al tuo stato d'animo, cosa preferisci generalmente?")
    preferenza_musica_per_mood = st.radio(
        "Seleziona una risposta:",
        [
            "Musica che rispecchia il mio stato d'animo",
            "Musica che mi aiuta a cambiare il mio stato d'animo",
            "Dipende dalla situazione",
            "Non scelgo musica in base al mio stato d'animo",
        ],
        index=None,
        key="final_preferenza_musica_per_mood",
    )

    # --------------------------------------------------------
    # SEZIONE 3: DATI SOCIO-DEMOGRAFICI
    # --------------------------------------------------------
    st.divider()
    st.header("3. Dati socio-demografici")

    genere_demografico = st.radio(
        "3.1 Indica il tuo genere:",
        ["Maschio", "Femmina", "Altro"],
        index=None,
        key="final_genere_demografico",
    )

    age_options = [str(age) for age in range(18, 80)] + ["80+"]
    eta = st.selectbox(
        "3.2 Indica la tua età:",
        age_options,
        index=None,
        placeholder="Seleziona l'età",
        key="final_eta",
    )

    region_options = [
        "Abruzzo", "Basilicata", "Calabria", "Campania", "Emilia-Romagna",
        "Friuli-Venezia Giulia", "Lazio", "Liguria", "Lombardia", "Marche",
        "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia", "Toscana",
        "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", "Veneto", "Vivo all'estero",
    ]
    regione_residenza = st.selectbox(
        "3.3 Indica la tua regione di residenza:",
        region_options,
        index=None,
        placeholder="Seleziona una regione",
        key="final_regione_residenza",
    )

    istruzione = st.selectbox(
        "3.4 Indica il tuo titolo di studio:",
        [
            "Terza media",
            "Diploma superiore",
            "Laurea triennale",
            "Laurea magistrale",
            "Dottorato di ricerca",
        ],
        index=None,
        placeholder="Seleziona un titolo di studio",
        key="final_istruzione",
    )

    st.divider()
    st.write(
        "Controlla le risposte e poi invia il questionario. "
        "Il commento sulla playlist è facoltativo."
    )
    invia_questionario = st.button(
        "Invia il questionario e termina 🎵",
        type="primary",
        use_container_width=True,
        key="invia_questionario_finale",
    )

    if invia_questionario:
        required_answers = {
            "gradimento della playlist": gradimento_playlist,
            "coerenza emotiva": coerenza_emotiva,
            "coerenza con i gusti musicali": gusti_musicali,
            "canzone preferita": canzone_preferita,
            "canzone meno preferita": canzone_meno_preferita,
            "numero di canzoni già conosciute": canzoni_gia_conosciute,
            "tempo medio di ascolto": frequenza_ascolto_musica,
            "modalità di scelta della musica": scelta_algoritmo,
            "preferenza tra musica conosciuta e nuova": scoperta_musica,
            "frequenza di scelta in base all'umore": playlist_mood,
            "preferenza musicale in base all'umore": preferenza_musica_per_mood,
            "genere": genere_demografico,
            "età": eta,
            "regione di residenza": regione_residenza,
            "titolo di studio": istruzione,
        }
        missing_answers = [
            label for label, value in required_answers.items() if value is None
        ]

        if missing_answers:
            st.error("Rispondi a tutte le domande obbligatorie prima di inviare il questionario.")
        else:
            try:
                with st.spinner("Salvataggio delle risposte in corso..."):
                    results_df, _ = load_results_from_github()
                    participant_id = generate_participant_id(results_df)
                    timestamp = datetime.now(ZoneInfo("Europe/Rome")).isoformat(timespec="seconds")
                    preferences = st.session_state.preferenze_iniziali

                    song_ids = [str(song["id"]).strip() for _, song in playlist_df.iterrows()]
                    song_id_columns = {
                        f"song_id_{index + 1}": song_ids[index] if index < len(song_ids) else pd.NA
                        for index in range(7)
                    }

                    # "si" solo se tutti i filtri iniziali sono stati rispettati.
                    requisiti_soddisfatti = (
                        "no" if st.session_state.playlist_fallback_used else "si"
                    )

                    # La modalità viene registrata solo quando la domanda è stata proposta.
                    if final_emotion in EMOTIONS_WITH_MODE_QUESTION:
                        coerenza_playlist = mode
                    else:
                        coerenza_playlist = pd.NA

                    new_row = {
                        "id_partecipante": participant_id,
                        "timestamp": timestamp,
                        "genere_preferito": preferences["genre"],
                        "lingua_preferita": preferences["language"],
                        "anno_inizio": preferences["start_year"],
                        "anno_fine": preferences["end_year"],
                        "preferenza_popolarita": preferences["popularity_preference"],
                        "emozione_dichiarata": st.session_state.emozione_dichiarata_iniziale,
                        "emozione_stimata": final_emotion,
                        "mood": pd.NA,
                        "requisiti_soddisfatti": requisiti_soddisfatti,
                        "coerenza_playlist": coerenza_playlist,
                        "numero_canzoni": len(playlist_df),
                        "gradimento_playlist": gradimento_playlist,
                        "coerenza_emotiva": coerenza_emotiva,
                        "gusti_musicali": gusti_musicali,
                        "canzone_preferita": canzone_preferita,
                        "canzone_meno_preferita": canzone_meno_preferita,
                        "canzoni_gia_conosciute": canzoni_gia_conosciute,
                        "commento_valutazione": (
                            commento_valutazione.strip() if commento_valutazione.strip() else pd.NA
                        ),
                        "frequenza_ascolto_musica": frequenza_ascolto_musica,
                        "scoperta_musica": scoperta_musica,
                        "playlist_mood": playlist_mood,
                        "scelta_algoritmo": scelta_algoritmo,
                        "preferenza_musica_per_mood": preferenza_musica_per_mood,
                        "genere": genere_demografico,
                        "eta": eta,
                        "regione_residenza": regione_residenza,
                        "istruzione": istruzione,
                    }
                    new_row.update(song_id_columns)
                    save_result_to_github(new_row)
                    st.session_state.risultato_salvato = True

                st.success("✅ Grazie! Le tue risposte sono state registrate correttamente.")
                st.write("Puoi chiudere questa pagina.")

            except RuntimeError as exc:
                if str(exc) == "CONFLICT_GITHUB":
                    st.error(
                        "Si è verificato un conflitto durante il salvataggio dei dati su GitHub. "
                        "Riprova tra qualche secondo."
                    )
                else:
                    st.error("Si è verificato un errore durante il salvataggio del questionario.")
                    st.exception(exc)
            except Exception as exc:
                st.error("Si è verificato un errore durante il salvataggio del questionario.")
                st.exception(exc)


# ============================================================
# MESSAGGIO FINALE DOPO IL SALVATAGGIO
# ============================================================

if st.session_state.risultato_salvato:
    st.divider()
    st.success("🎉 Questionario completato! Grazie per aver partecipato al progetto.")
