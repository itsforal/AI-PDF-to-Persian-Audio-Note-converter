# AI-Powered Lecture Note & Audiobook Generator

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![AI](https://img.shields.io/badge/GenAI-Gemini%20%26%20DeepSeek-orange)

## Overview
This project implements an **experimental pipeline** to automate the conversion of academic PDF textbooks (containing math and diagrams) into **structured lecture notes** in Persian (Farsi), and subsequently generates audio versions.

The tool aims to **streamline the study process** by bridging the gap between hard-copy academic material and digital accessibility tools.

## Key Features
1.  **Visual Extraction Pipeline**: Utilizes **Google Gemini Pro Vision** to parse text and \LaTeX{} math formulas directly from rendered PDF page images.
2.  **Content Refinement**: Integrates **DeepSeek V3** to rewrite raw extractions into coherent, academic Persian summaries.
3.  **Resumable Workflow**: Implements state management (via JSON logs) to handle interruptions during long processing tasks.
4.  **Audio Synthesis**: Converts the final Markdown notes into audio files using **Edge-TTS**, supporting bilingual context.

## System Architecture
- **Input**: PDF Document.
- **Stage 1 (Rendering)**: `PyMuPDF` converts pages to High-DPI images for vision analysis.
- **Stage 2 (Vision)**: `Gemini API` extracts visual data and mathematical notation.
- **Stage 3 (Refinement)**: `DeepSeek API` structures the content for readability.
- **Stage 4 (Audio)**: `EdgeTTS` + `PyDub` synthesize and merge audio segments.
