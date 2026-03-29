from __future__ import annotations

import hashlib

import streamlit as st

from chat_store import append_exchange, create_chat, delete_chat, load_store, rename_chat
from rag import ask
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


def _render_sources(sources: list[str]) -> None:
    if not sources:
        return

    top_source = sources[0]
    st.markdown(f"**Top source**: {top_source}")
    if len(sources) > 1:
        with st.expander("More sources", expanded=False):
            for source in sources[1:]:
                st.write(source)


st.set_page_config(
    page_title="Voice-Enabled RAG Assistant",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --page-bg: linear-gradient(180deg, #f7f4ef 0%, #f3efe8 46%, #ece6db 100%);
            --card-bg: rgba(255, 255, 255, 0.72);
            --card-border: rgba(94, 74, 47, 0.14);
            --ink: #1f2937;
            --muted: #6b7280;
            --accent: #8a5a2b;
            --accent-soft: rgba(138, 90, 43, 0.12);
        }

        .stApp {
            background: var(--page-bg);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5efe5 0%, #efe7da 100%);
            border-right: 1px solid rgba(94, 74, 47, 0.12);
        }

        section.main > div {
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 21rem;
        }

        .hero-wrap {
            display: grid;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .eyebrow {
            font-size: 0.82rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent);
            font-weight: 700;
        }

        .hero-card, .info-card, .voice-shell {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 22px;
            box-shadow: 0 18px 40px rgba(31, 41, 55, 0.08);
            backdrop-filter: blur(10px);
        }

        .hero-card {
            padding: 1.3rem 1.4rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            color: var(--ink);
            margin: 0;
            line-height: 1.05;
        }

        .hero-subtitle {
            margin: 0.45rem 0 0;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.5;
        }

        .stat-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.5rem 0.8rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--ink);
            font-size: 0.88rem;
            font-weight: 600;
            margin-right: 0.5rem;
            margin-top: 0.5rem;
        }

        .info-card {
            padding: 1rem 1.05rem;
        }

        .info-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }

        .info-value {
            color: var(--ink);
            font-weight: 700;
            font-size: 1.05rem;
        }

        .voice-shell {
            padding: 1rem 1rem 0.9rem;
        }

        .voice-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.75rem;
        }

        .voice-title {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--ink);
        }

        .voice-hint {
            margin: 0.2rem 0 0;
            color: var(--muted);
            font-size: 0.88rem;
        }

        section.main div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stAudioInput"]) {
            position: fixed !important;
            bottom: 0 !important;
            left: 50% !important;
            transform: translateX(-50%);
            width: min(1180px, calc(100vw - 1.5rem)) !important;
            z-index: 999 !important;
            background: transparent;
            border-top: none;
            box-shadow: none;
        }

        section.main div[data-testid="stVerticalBlock"] > div:last-child {
            position: fixed !important;
            bottom: 0 !important;
            left: 50% !important;
            transform: translateX(-50%);
            width: min(1180px, calc(100vw - 1.5rem)) !important;
            z-index: 999 !important;
        }

        section.main > div {
            padding-bottom: 22rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-wrap">
      <div class="hero-card">
        <div class="eyebrow">Voice RAG Workspace</div>
        <h1 class="hero-title">Voice-Enabled RAG Assistant</h1>
        <p class="hero-subtitle">Record a question or upload an audio clip, and get a grounded answer from the vector knowledge base with spoken output.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    st.markdown("### Workspace")
    st.caption("Your chats, voice settings, and document context live here.")
    if st.button("New Chat", type="primary", use_container_width=True):
        new_chat_id = create_chat(store, "New Chat")
        st.session_state.store = load_store()
        st.session_state.active_chat_id = new_chat_id
        st.session_state.last_audio_digest = ""
        st.session_state.last_audio_source = ""
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
            st.session_state.last_audio_source = ""
            st.session_state.last_transcript = ""
            st.session_state.last_error = ""
            st.rerun()

    st.divider()
    st.markdown("### Voice Mode")
    voice_language = st.selectbox(
        "Choose language",
        options=["Auto", "English", "Bangla"],
        index=0,
        help="Auto works for mixed input. Pick English or Bangla if you know the language upfront.",
    )
    if voice_language == "Bangla":
        st.info("Bangla transcription is enabled. Spoken output depends on the voices installed on Windows.")
    else:
        st.caption("Tip: shorter WAV clips usually transcribe fastest.")

current_chat = st.session_state.store["chats"][st.session_state.active_chat_id]

chat_col = st.container()
with chat_col:
    if current_chat.get("messages"):
        for message in current_chat.get("messages", []):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant":
                    sources = message.get("sources", [])
                    _render_sources(sources)
    else:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-label">Start here</div>
                <div class="info-value">Record a voice prompt or upload an audio file using the dock at the bottom.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

transcript = ""

if "last_audio_digest" not in st.session_state:
    st.session_state.last_audio_digest = ""
if "last_audio_source" not in st.session_state:
    st.session_state.last_audio_source = ""
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "last_answer_audio" not in st.session_state:
    st.session_state.last_answer_audio = {}
if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}
if "input_widget_nonce" not in st.session_state:
    st.session_state.input_widget_nonce = 0


def _reset_voice_inputs() -> None:
    st.session_state.input_widget_nonce += 1
    st.session_state.audio_input_mode = "Record voice"
    st.session_state.last_audio_digest = ""
    st.session_state.last_audio_source = ""
    st.session_state.last_transcript = ""
    st.session_state.last_error = ""

footer = st.container()
with footer:
    st.markdown(
        """
        <div class="voice-shell">
          <div class="voice-header">
            <div>
              <p class="voice-title">Voice Dock</p>
              <p class="voice-hint">Record or upload audio here. The assistant will transcribe and answer automatically.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "audio_input_mode" not in st.session_state:
        st.session_state.audio_input_mode = "Record voice"

    input_mode = st.radio(
        "Input source",
        options=["Record voice", "Upload audio"],
        index=0 if st.session_state.audio_input_mode == "Record voice" else 1,
        horizontal=True,
        key="audio_input_mode",
    )

    if input_mode == "Record voice":
        source_audio = st.audio_input("Record your voice", key=f"voice_input_{st.session_state.input_widget_nonce}")
    else:
        source_audio = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            help="Whisper runs locally. WAV is fastest. MP3 and M4A need ffmpeg.",
            key=f"audio_file_{st.session_state.input_widget_nonce}",
        )

    if source_audio is not None:
        audio_bytes, audio_name = _audio_payload(source_audio)
        audio_digest = hashlib.sha1(audio_bytes).hexdigest() if audio_bytes else ""
        audio_source_type = "voice_input" if input_mode == "Record voice" else "file_upload"
        current_audio_source = f"{audio_source_type}:{audio_name}:{voice_language}"
        same_audio_source = st.session_state.last_audio_source == current_audio_source

        st.audio(audio_bytes)
        st.caption("Audio received. Transcribing now, then the assistant will speak the answer.")

        cached_transcript = ""
        if same_audio_source and audio_digest and st.session_state.last_audio_digest == audio_digest:
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
                    st.session_state.last_audio_source = current_audio_source
                    st.session_state.last_transcript = transcript
                    st.session_state.last_error = ""
                    st.success("Transcription complete")
                    transcript_card = st.container(border=True)
                    with transcript_card:
                        st.markdown("**Recognized text**")
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

                    answer_card = st.container(border=True)
                    with answer_card:
                        st.markdown("**Assistant response**")
                        st.write(answer)
                    audio_output = st.session_state.last_answer_audio.get(answer, b"")
                    if audio_output:
                        st.caption("Speaking the answer now")
                        st.audio(audio_output, format="audio/wav")
                    _render_sources(sources)
                    _reset_voice_inputs()
                    st.rerun()
                else:
                    st.session_state.last_audio_digest = audio_digest
                    st.session_state.last_audio_source = current_audio_source
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
