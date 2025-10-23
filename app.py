# -*- coding: utf-8 -*-
"""
Meeting Audio Transcriber + Summarizer
--------------------------------------
Transcribes meeting audio using OpenAI Whisper (local) and summarizes it using Google's Gemini.
Source: https://colab.research.google.com/drive/1gswOcGYOLAXqBjTCLiCGlH1buaicaX_v
License: MIT
"""

# === Setup ===
!pip install -q transformers torch torchaudio librosa google-generativeai

import os
import torch
import librosa
import google.generativeai as genai
from transformers import pipeline
from google.colab import files, userdata
from IPython.display import display, Markdown
import ipywidgets as widgets

# === Configure Google API Key ===
try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google API Key configured successfully!")
except userdata.SecretNotFoundError:
    print('❌ ERROR: Secret "GOOGLE_API_KEY" not found.')
    print('➡️ Go to the "Secrets" tab (key icon) and add your Google API key.')
except Exception as e:
    print(f"⚠️ An error occurred: {e}")

# === Load Whisper Model ===
print("\n⏳ Loading local Whisper model... (this may take a few minutes)")
try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small.en",  # Optimized lightweight model
        device=device,
    )
    print(f"✅ Whisper model loaded successfully on device: {device}")
except Exception as e:
    print(f"❌ Could not load Whisper model: {e}")
    transcriber = None


# === Core Function: Audio Processing ===
def process_audio(audio_path):
    """Transcribe and summarize an uploaded meeting audio file."""
    if transcriber is None:
        print("⚠️ Transcription model unavailable.")
        return None, None

    try:
        print("\n🎧 Starting transcription...")

        # Load and transcribe
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        transcription_result = transcriber(audio_input, return_timestamps=True)
        transcript = transcription_result["text"]
        print("✅ Transcription complete.")

        # Summarize via Gemini
        print("🧠 Starting summarization...")
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        Based on the following meeting transcript, provide a structured summary including:
        - **Key Decisions**: Bulleted list of important decisions made.
        - **Action Items**: Bulleted list of assigned tasks with responsible persons.

        Transcript:
        ---
        {transcript}
        ---
        """

        response = model.generate_content(prompt)
        summary = response.text
        print("✅ Summarization complete.")

        return transcript, summary

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return None, None

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# === File Upload Widget (Colab UI) ===
print("\n--- Please upload your meeting audio file ---")
uploader = widgets.FileUpload(accept='audio/*', multiple=False)

def on_upload(change):
    if not change.new:
        return

    for file_name, uploaded_file_info in change.new.items():
        with open(file_name, 'wb') as f:
            f.write(uploaded_file_info['content'])

        print(f"\n📁 File '{file_name}' uploaded successfully. Starting processing...")

        transcript, summary = process_audio(file_name)

        if transcript:
            display(Markdown("---"))
            display(Markdown("## 📜 Transcription"))
            display(Markdown(transcript))
        if summary:
            display(Markdown("## ✨ Summary & Action Items"))
            display(Markdown(summary))

    uploader._counter = 0

uploader.observe(on_upload, names='value')
display(uploader)


# === List Available Gemini Models ===
try:
    print("\n🧩 Available Gemini Models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"⚠️ Could not list models: {e}")
