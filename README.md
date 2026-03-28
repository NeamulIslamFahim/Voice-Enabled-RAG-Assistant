# Voice-Enabled RAG Assistant

A local Streamlit app that accepts voice input or uploaded audio, transcribes it with Whisper, searches a bundled knowledge base, and returns the answer in both text and audio.

## Features

- Voice input with `st.audio_input`
- Voice file upload support
- Local transcription with Whisper
- Faster CPU transcription via `faster-whisper`
- Text-to-speech response audio
- Conversation history in the sidebar
- New chat, rename chat, and delete chat support
- Bangla language support for transcription

## Project Structure

- `app.py` - Streamlit UI and app flow
- `rag.py` - Retrieval and answer formatting
- `whisper_stt.py` - Speech-to-text helper
- `tts_speech.py` - Text-to-speech helper
- `chat_store.py` - Saved chat history management
- `faiss_index/` - Bundled document index
- `requirements.txt` - Python dependencies

## Model Training

The model training was completed in Google Colab, and the training files and artifacts are stored in Google Drive.

- Google Drive folder: https://drive.google.com/drive/folders/1l8hP3l9vRWArtNQZa_k5-ecprx0VAERD?usp=sharing

### Explanation

- Google Colab was used for the training workflow because it provides an easy cloud notebook environment with access to GPUs and a simple way to manage experiment files.
- The Google Drive folder contains the saved training resources, outputs, and related project materials.
- This GitHub repository focuses on the voice-enabled RAG assistant application itself, while the training assets are kept in Drive for easier organization and sharing.

## Requirements

### Python

- Python 3.12 is recommended

### Optional but recommended

- `ffmpeg` for non-WAV audio formats such as MP3, M4A, OGG, and FLAC

### Notes

- WAV voice files work best and avoid extra conversion overhead
- On Windows, the first model download can take a while
- The app uses local models and does not require a remote API key

## Setup

### 1. Clone or open the project

Open the project folder in your terminal:

```powershell
cd "C:\Users\FAHIM\OneDrive\Desktop\Voice-Enabled RAG Assistant"
```

### 2. Create and activate a virtual environment

Recommended on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer a user install instead of a virtual environment:

```powershell
python -m pip install --user -r requirements.txt
```

## ffmpeg Setup

`ffmpeg` is required if you want to upload or process non-WAV audio files.

### Install with Chocolatey

```powershell
choco install ffmpeg
```

### Install manually

1. Download ffmpeg for Windows
2. Extract it somewhere permanent
3. Add the `bin` folder to your `PATH`
4. Reopen PowerShell

### Verify installation

```powershell
ffmpeg -version
```

## Run the App

Start Streamlit:

```powershell
python -m streamlit run app.py
```

Then open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

### Voice input

1. Choose a voice language from the sidebar:
   - `Auto`
   - `English`
   - `Bangla`
2. Record your voice using `Record your voice`
3. Click `Transcribe Voice`
4. The assistant will show:
   - transcribed text
   - the answer
   - audio playback of the answer

### Voice file upload

1. Upload a voice file using `Or upload a voice file`
2. Click `Transcribe Voice`
3. The app will transcribe and answer from the indexed document

### Conversation history

- Chats appear in the sidebar
- Click `New Chat` to start a fresh conversation
- Use the selected chat title box to rename a conversation
- Use `Delete` to remove a chat

## Bangla Support

The app supports Bangla transcription.

- Select `Bangla` in the sidebar
- Use Bangla speech input or Bangla audio files
- For best results, use clear, close microphone audio

Note:
- The assistant can transcribe Bangla
- The spoken audio response depends on the voices installed on Windows
- Bangla text output may still be more natural than Bangla speech output on a default Windows setup

## Text-to-Speech Output

The assistant response is shown in text and also converted to audio.

- TTS is handled locally with `pyttsx3`
- The generated audio is cached so repeat responses are faster
- If TTS is unavailable, the app still shows the text response

## Troubleshooting

### `FP16 is not supported on CPU`

This has already been handled in the code by forcing CPU-safe transcription settings.

### `ffmpeg` not found

Install `ffmpeg` and add it to your `PATH`.

### Audio transcription is slow

Try these:

- Use WAV files instead of MP3 or M4A
- Keep the default `tiny.en` model for English
- Use `Bangla` only when you need Bangla transcription
- Close other heavy Python apps if your CPU is busy

### Hugging Face symlink warning on Windows

This is already disabled in the app code. If needed, you can also set:

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

### `WinError 32` during install

This usually means a Python process is holding a file open.

Try:

1. Close Streamlit
2. Close other Python shells or editors
3. Run the install again
4. Prefer a virtual environment for future installs

## Dependency List

- `streamlit`
- `openai-whisper`
- `faster-whisper`
- `pyttsx3`

## Recommended Workflow

1. Activate the virtual environment
2. Start the app
3. Record or upload voice
4. Choose the language
5. Click `Transcribe Voice`
6. Review the text answer and audio response

## License

Add your preferred license here if you want to publish or share the project.
