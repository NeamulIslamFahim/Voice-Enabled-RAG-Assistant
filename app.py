from __future__ import annotations

import hashlib

import streamlit as st

from chat_store import append_exchange, create_chat, delete_chat, load_store, rename_chat
from rag import ask, get_document_count
from whisper_stt import transcribe_audio
from tts_speech import text_to_speech_bytes


def _audio_payload(source):
    if source is None:
        return b"", "voice.wav"
    if hasattr(source, "getvalue"):
        data = source.getvalue()
    elif hasattr(source, "read"):
        data = source.read()
    elif isinstance(source, bytes):
        data = source
    else:
        data = bytes(source)
    name = str(getattr(source, "name", "voice.wav"))
    return data, name


def _sorted_chat_ids(store):
    chats = store.get("chats", {})
    return sorted(
        chats.keys(),
        key=lambda chat_id: chats[chat_id].get("updated_at", ""),
        reverse=True,
    )


def _chat_label(chat):
    title = chat.get("title", "New Chat")
    updated_at = chat.get("updated_at", "")[:19].replace("T", " ")
    return f"{title}  ({updated_at})" if updated_at else title


st.set_page_config(
    page_title="Voice-Enabled RAG Assistant",
    layout="centered",
)

st.markdown(
    """
    <style>
        section.main div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stAudioInput"]) {
            position: fixed !important;
            bottom: 0 !important;
            left: 50% !important;
            transform: translateX(-50%);
            width: min(740px, calc(100vw - 2rem)) !important;
            z-index: 999 !important;
            background: var(--background-color);
            border-top: 1px solid rgba(49, 51, 63, 0.12);
            box-shadow: 0 -8px 24px rgba(0, 0, 0, 0.08);
            padding: 1rem 1rem 0.9rem;
            border-radius: 16px 16px 0 0;
        }

        section.main div[data-testid="stVerticalBlock"] > div:last-child {
            position: fixed !important;
            bottom: 0 !important;
            left: 50% !important;
            transform: translateX(-50%);
            width: min(740px, calc(100vw - 2rem)) !important;
            z-index: 999 !important;
        }

        section.main > div {
            padding-bottom: 20rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Voice Enabled RAG Assistant")
st.caption("Speak your question or upload a voice file.")

if "store" not in st.session_state:
    st.session_state.store = load_store()

store = st.session_state.store
if not store.get("chats"):
    create_chat(store, "New Chat")
    store = load_store()
    st.session_state.store = store

chat_ids = _sorted_chat_ids(store)
if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in store["chats"]:
    st.session_state.active_chat_id = store.get("active_chat_id") or chat_ids[0]

with st.sidebar:
    st.header("Conversation History")
    if st.button("New Chat", type="primary", use_container_width=True):
        new_chat_id = create_chat(store, "New Chat")
        st.session_state.store = load_store()
        st.session_state.active_chat_id = new_chat_id
        st.session_state.last_audio_digest = ""
        st.session_state.last_transcript = ""
        st.session_state.last_error = ""
        st.rerun()

    chat_ids = _sorted_chat_ids(st.session_state.store)
    active_chat_id = st.radio(
        "Saved chats",
        options=chat_ids,
        index=chat_ids.index(st.session_state.active_chat_id),
        format_func=lambda chat_id: _chat_label(st.session_state.store["chats"][chat_id]),
        label_visibility="collapsed",
    )
    st.session_state.active_chat_id = active_chat_id

    rename_key = f"rename_title_{active_chat_id}"
    if rename_key not in st.session_state:
        st.session_state[rename_key] = st.session_state.store["chats"][active_chat_id].get("title", "New Chat")

    st.text_input("Chat title", key=rename_key)
    rename_col, delete_col = st.columns(2)
    with rename_col:
        if st.button("Rename", use_container_width=True):
            rename_chat(st.session_state.store, active_chat_id, st.session_state[rename_key])
            st.session_state.store = load_store()
            st.rerun()
    with delete_col:
        if st.button("Delete", use_container_width=True):
            new_active_chat_id = delete_chat(st.session_state.store, active_chat_id)
            st.session_state.store = load_store()
            st.session_state.active_chat_id = new_active_chat_id
            st.session_state.last_audio_digest = ""
            st.session_state.last_transcript = ""
            st.session_state.last_error = ""
            st.rerun()

    st.divider()
    voice_language = st.selectbox(
        "Voice language",
        options=["Auto", "English", "Bangla"],
        index=0,
    )
    if voice_language == "Bangla":
        st.caption("Bangla transcription is enabled. Audio output depends on the voices installed on Windows.")

current_chat = st.session_state.store["chats"][st.session_state.active_chat_id]

for message in current_chat.get("messages", []):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            sources = message.get("sources", [])
            if sources:
                with st.expander("Sources", expanded=False):
                    for source in sources:
                        st.write(source)

transcript = ""

if "last_audio_digest" not in st.session_state:
    st.session_state.last_audio_digest = ""
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "last_answer_audio" not in st.session_state:
    st.session_state.last_answer_audio = {}
if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}

footer = st.container()
with footer:
    st.divider()
    st.subheader("Voice Input")

    voice_col, file_col = st.columns([1, 1])
    with voice_col:
        voice_input = st.audio_input("Record your voice")
    with file_col:
        audio_file = st.file_uploader(
            "Or upload a voice file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            help="Whisper runs locally. MP3 and M4A inputs require ffmpeg to be installed on your machine.",
        )

    source_audio = voice_input if voice_input is not None else audio_file

    if source_audio is not None:
        audio_bytes, audio_name = _audio_payload(source_audio)
        audio_digest = hashlib.sha1(audio_bytes).hexdigest() if audio_bytes else ""

        st.audio(audio_bytes)
        st.caption("Voice is ready. It will be transcribed automatically. WAV files are usually the fastest.")

        cached_transcript = ""
        if audio_digest and st.session_state.last_audio_digest == audio_digest:
            cached_transcript = st.session_state.last_transcript

        if cached_transcript:
            transcript = cached_transcript
            st.success("Loaded cached transcription.")
            st.write("Recognized text")
            st.write(transcript)
        elif audio_bytes:
            try:
                selected_language = {
                    "Auto": "auto",
                    "English": "en",
                    "Bangla": "bn",
                }[voice_language]
                model_name = "small" if selected_language in {"auto", "bn"} else "small.en"
                with st.spinner("Transcribing voice..."):
                    transcript = transcribe_audio(
                        audio_bytes,
                        audio_name=audio_name,
                        model_name=model_name,
                        language=selected_language,
                    )

                if transcript:
                    st.session_state.last_audio_digest = audio_digest
                    st.session_state.last_transcript = transcript
                    st.session_state.last_error = ""
                    st.success("Audio transcribed successfully.")
                    st.write("Recognized text")
                    st.write(transcript)

                    cache_key = transcript.strip()
                    if cache_key in st.session_state.qa_cache:
                        answer, sources = st.session_state.qa_cache[cache_key]
                    else:
                        with st.spinner("Searching the knowledge base..."):
                            answer, sources = ask(cache_key)
                        st.session_state.qa_cache[cache_key] = (answer, sources)

                    if answer not in st.session_state.last_answer_audio:
                        try:
                            with st.spinner("Generating audio response..."):
                                st.session_state.last_answer_audio[answer] = text_to_speech_bytes(answer)
                        except Exception as exc:
                            st.session_state.last_answer_audio[answer] = b""
                            st.caption(f"Audio response unavailable: {exc}")

                    append_exchange(
                        st.session_state.store,
                        st.session_state.active_chat_id,
                        cache_key,
                        answer,
                        sources,
                    )
                    st.session_state.store = load_store()

                    st.chat_message("assistant").write(answer)
                    audio_output = st.session_state.last_answer_audio.get(answer, b"")
                    if audio_output:
                        st.caption("Speaking the answer now.")
                        st.audio(audio_output, format="audio/wav")
                    if sources:
                        with st.expander("Sources", expanded=False):
                            for source in sources:
                                st.write(source)
                else:
                    st.session_state.last_audio_digest = audio_digest
                    st.session_state.last_transcript = ""
                    st.warning("The audio file was processed, but no text was returned.")
            except Exception as exc:
                st.session_state.last_error = str(exc)
                st.error(f"Audio transcription failed: {exc}")
        elif st.session_state.last_error and audio_digest == st.session_state.last_audio_digest:
            st.error(f"Audio transcription failed: {st.session_state.last_error}")
        elif voice_language == "Bangla":
            st.caption("Bangla voice input is enabled. Use a Bangla audio file or speech input for best results.")
    else:
        st.info("Record your voice or upload an audio file to begin.")
