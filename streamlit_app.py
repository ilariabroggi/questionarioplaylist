import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
import json
from io import BytesIO
from pathlib import Path

# ============================================================
# CONFIGURAZIONE DELLA PAGINA
# ============================================================

st.set_page_config(
    page_title="Playlist ed emozioni",
    page_icon="🎵",
    layout="centered"
)


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
# FUNZIONE PER ESTRARRE LE FEATURE MFCC
# ============================================================

def extract_mfcc_features(audio_file):

    # Leggiamo l'audio direttamente dalla registrazione
    audio_bytes = audio_file.getvalue()

    signal, sample_rate = librosa.load(
        BytesIO(audio_bytes),
        sr=None
    )

    # Estrazione dei 13 MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=13
    )

    # Media dei 13 MFCC
    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    # Deviazione standard dei 13 MFCC
    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    # Unione media + deviazione standard
    feature_vector = np.concatenate(
        [mfcc_mean, mfcc_std]
    )

    # Creazione DataFrame
    features_df = pd.DataFrame(
        [feature_vector],
        columns=feature_cols
    )

    return features_df


# ============================================================
# FUNZIONE PER CLASSIFICARE L'EMOZIONE
# ============================================================

def predict_emotion(features_df):

    # Standardizzazione utilizzando
    # lo stesso scaler utilizzato su RAVDESS
    features_std = scaler.transform(
        features_df[feature_cols]
    )

    # DataFrame standardizzato
    features_std_df = pd.DataFrame(
        features_std,
        columns=feature_cols
    )

    # Calcolo delle distanze euclidee
    distances = {}

    for emotion in centroids.index:

        centroid = centroids.loc[
            emotion,
            feature_cols
        ].values

        distance = np.linalg.norm(
            features_std_df.iloc[0].values - centroid
        )

        distances[emotion] = distance

    # Emozione corrispondente alla distanza minima
    predicted_emotion = min(
        distances,
        key=distances.get
    )

    return predicted_emotion, distances


# ============================================================
# TITOLO
# ============================================================

st.title("🎵 Emozioni, voce e musica")

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

st.header("🎧 Le tue preferenze musicali")

st.write("""
Prima di procedere con la registrazione della voce, rispondi alle seguenti
domande sulle tue preferenze musicali.
""")


genere = st.radio(
    "Qual è il tuo genere musicale preferito?",
    [
        "Pop",
        "Rock",
        "Rap / Hip-Hop",
        "Techno / Musica elettronica",
        "Musica classica"
    ]
)


lingua = st.radio(
    "In quale lingua preferisci ascoltare musica?",
    [
        "Italiano",
        "Inglese",
        "Spagnolo / Lingue latino-americane"
    ]
)


periodo = st.radio(
    "Quale periodo musicale ascolti maggiormente?",
    [
        "Dal 1970 al 1995",
        "Dal 1996 al 2010",
        "Dal 2011 a oggi"
    ]
)


st.divider()


# ============================================================
# EMOZIONE DICHIARATA
# ============================================================

st.header("💭 Come ti senti in questo momento?")

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
        "Neutral",
        "Calm",
        "Happy",
        "Sad",
        "Angry",
        "Fearful",
        "Disgust",
        "Surprised"
    ]
)


st.divider()


# ============================================================
# REGISTRAZIONE DELLA VOCE
# ============================================================

st.header("🎙️ Registrazione della voce")

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

st.subheader("Registrazione 1")

st.write(
    "Pronuncia la seguente frase:"
)

st.info(
    "🗣️ «I bambini stanno parlando vicino alla porta.»"
)

audio_1 = st.audio_input(
    "Registra la prima frase"
)


# ============================================================
# SECONDA REGISTRAZIONE
# ============================================================

st.subheader("Registrazione 2")

st.write(
    "Pronuncia la seguente frase:"
)

st.info(
    "🗣️ «I cani sono seduti vicino alla porta.»"
)

audio_2 = st.audio_input(
    "Registra la seconda frase"
)


st.divider()


# ============================================================
# RIEPILOGO
# ============================================================

st.header("📋 Riepilogo")

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
    f"**Periodo musicale preferito:** {periodo}"
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

    if audio_1 is None or audio_2 is None:

        st.warning(
            "⚠️ Per continuare è necessario registrare entrambe le frasi."
        )

    else:

        with st.spinner(
            "Analisi della voce in corso..."
        ):

            # ------------------------------------------------
            # ESTRAZIONE DELLE FEATURE
            # ------------------------------------------------

            features_1 = extract_mfcc_features(
                audio_1
            )

            features_2 = extract_mfcc_features(
                audio_2
            )


            # ------------------------------------------------
            # CLASSIFICAZIONE DELLE DUE REGISTRAZIONI
            # ------------------------------------------------

            emotion_1, distances_1 = predict_emotion(
                features_1
            )

            emotion_2, distances_2 = predict_emotion(
                features_2
            )


            # ------------------------------------------------
            # EMOZIONE FINALE
            # ------------------------------------------------

            if emotion_1 == emotion_2:

                final_emotion = emotion_1

            else:

                # Se le due registrazioni producono
                # emozioni differenti, scegliamo quella
                # con la distanza minore dal proprio centroide

                min_distance_1 = min(
                    distances_1.values()
                )

                min_distance_2 = min(
                    distances_2.values()
                )

                if min_distance_1 <= min_distance_2:

                    final_emotion = emotion_1

                else:

                    final_emotion = emotion_2


        # ====================================================
        # RISULTATO
        # ====================================================

        st.success(
            "Analisi completata! 🎉"
        )

        st.header(
            "🧠 Emozione stimata"
        )

        st.write(
            f"## {final_emotion}"
        )


        # ----------------------------------------------------
        # CONFRONTO CON EMOZIONE DICHIARATA
        # ----------------------------------------------------

        st.subheader(
            "Confronto con la tua risposta"
        )

        if final_emotion == emozione_dichiarata:

            st.success(
                f"L'emozione dichiarata ({emozione_dichiarata}) "
                f"coincide con quella stimata dalla voce."
            )

        else:

            st.info(
                f"Hai indicato **{emozione_dichiarata}**, "
                f"mentre l'analisi della voce ha stimato "
                f"**{final_emotion}**."
            )


        # ----------------------------------------------------
        # RISULTATI DELLE DUE REGISTRAZIONI
        # ----------------------------------------------------

        with st.expander(
            "Visualizza i risultati delle due registrazioni"
        ):

            st.write(
                f"**Registrazione 1:** {emotion_1}"
            )

            st.write(
                f"**Registrazione 2:** {emotion_2}"
            )

            st.write(
                "Le distanze dai centroidi sono state calcolate "
                "nello spazio delle 26 feature MFCC standardizzate."
            )


        # ====================================================
        # PROSSIMO PASSAGGIO
        # ====================================================

        st.divider()

        st.info(
            "🎵 La generazione della playlist verrà aggiunta "
            "nella fase successiva."
        )
        