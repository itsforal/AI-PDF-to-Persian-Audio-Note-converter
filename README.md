# AI-Powered Lecture Note & Audiobook Generator

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![AI](https://img.shields.io/badge/GenAI-Gemini%20%26%20DeepSeek-orange)

## 📄 Overview
This tool automates the conversion of complex academic PDF textbooks (containing math and diagrams) into high-quality, pedagogical lecture notes in **Persian (Farsi)**, followed by generating an audio version (Audiobook).

It solves the problem of digitizing and studying hard-copy academic material by leveraging a multi-model AI pipeline.

## 🚀 Features
1.  **Optical Character Recognition (OCR) with Vision AI**: Uses **Google Gemini Pro Vision** to extract text and LaTeX math formulas from PDF images.
2.  **Pedagogical Refinement**: Uses **DeepSeek V3** to rewrite raw extractions into structured, academic Persian lecture notes.
3.  **State Management**: Tracks progress (`process_log.json`) to resume operations in case of interruptions.
4.  **Audiobook Generation**: Converts the final Markdown notes into an MP3 file using **Edge-TTS** (Neural voice).

## 🛠️ Architecture
- **Input**: PDF Document.
- **Stage 1**: `PyMuPDF` renders pages to High-DPI images.
- **Stage 2**: `Gemini API` performs visual extraction (Text + LaTeX).
- **Stage 3**: `DeepSeek API` contextually summarizes and translates to Persian.
- **Stage 4**: `EdgeTTS` + `PyDub` synthesizes and merges audio segments.

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/lecture-note-generator.git
   cd lecture-note-generator