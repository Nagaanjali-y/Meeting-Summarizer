# 🎙️ Meeting Audio Summarizer (for Google Colab)

This project provides a Google Colab notebook to transcribe and summarize meeting audio. It uses a local Whisper model for high-quality transcription and the Google Gemini API for generating structured summaries of key decisions and action items.

---

## ✨ Features

-   **High-Quality Transcription**: Utilizes OpenAI's `whisper-small.en` model running directly on Colab's free GPU for fast and accurate speech-to-text.
-   **Intelligent Summarization**: Leverages Google's `gemini-1.5-flash-latest` model to create structured summaries.
-   **Interactive UI**: Uses `ipywidgets` or `gradio` to provide an easy file-upload interface directly within the notebook.
-   **Zero Local Setup**: Runs entirely in the cloud with Google Colab, requiring no local installation of complex libraries.

---

## 🛠️ Technologies Used

-   **Environment**: Google Colab (with T4 GPU)
-   **Python**
-   **Transcription**: Hugging Face Transformers with `openai/whisper-small.en`
-   **Summarization**: Google Gemini API
-   **Audio Processing**: Librosa
-   **Core AI Framework**: PyTorch

---

## 🚀 How to Use

1.  **Open in Google Colab:**

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YourUsername/Meeting-Summarizer-Colab/blob/main/app.py)

2.  **Add API Key**:
    * On the left side of the Colab notebook, click the **key icon** 🔑 to open the Secrets manager.
    * Create a new secret named `GOOGLE_API_KEY` and paste your Google AI API key as the value.

3.  **Enable GPU**:
    * Go to **Runtime** -> **Change runtime type**.
    * Select **T4 GPU** from the "Hardware accelerator" dropdown and click **Save**.

4.  **Run the Notebook**:
    * Click **Runtime** -> **Run all**.
    * The notebook will install all necessary libraries, load the models, and display a file upload button.

5.  **Upload Audio**:
    * Use the file upload button to select your meeting audio file. The transcription and summary will be displayed in the output.
