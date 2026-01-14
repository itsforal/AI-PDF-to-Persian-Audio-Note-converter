import os
import time
import json
import io
import re
import asyncio
import nest_asyncio
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path 
import google.generativeai as genai
from openai import OpenAI
import edge_tts
from pydub import AudioSegment
from dotenv import load_dotenv
from typing import Optional, Dict, List

# Apply nested asyncio for Jupyter/IDE compatibility
nest_asyncio.apply()

# Load environment variables for security
load_dotenv()

class Config:
    """Configuration settings for the application."""
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    
    # Paths
    BASE_DIR = Path("output")
    PDF_SOURCE = Path("input_document.pdf")  # CHANGE THIS TO YOUR PDF PATH
    
    # Output Files
    OUTPUT_TEXT = BASE_DIR / "Master_Lecture_Notes.md"
    OUTPUT_AUDIO = BASE_DIR / "Audiobook.mp3"
    LOG_FILE = BASE_DIR / "process_log.json"
    
    # Settings
    DPI = 300  # High resolution for math detection
    TTS_VOICE = "fa-IR-DilaraNeural" # Microsoft Edge Neural Voice
    MAX_RETRIES = 3

    @staticmethod
    def ensure_directories():
        """Creates necessary output directories."""
        Config.BASE_DIR.mkdir(parents=True, exist_ok=True)

class AIProcessor:
    """Handles interactions with AI Models (Gemini & DeepSeek)."""
    
    def __init__(self):
        self._setup_gemini()
        self._setup_deepseek()

    def _setup_gemini(self):
        if not Config.GEMINI_KEY:
            raise ValueError("Gemini API Key missing in .env file")
        genai.configure(api_key=Config.GEMINI_KEY)
        self.vision_model = self._select_best_gemini_model()

    def _setup_deepseek(self):
        if not Config.DEEPSEEK_KEY:
            raise ValueError("DeepSeek API Key missing in .env file")
        self.ds_client = OpenAI(
            api_key=Config.DEEPSEEK_KEY, 
            base_url="https://api.deepseek.com"
        )

    def _select_best_gemini_model(self):
        """Dynamically selects the most capable available vision model."""
        try:
            available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # Preference list: Newer models first
            preferences = [
                'models/gemini-1.5-pro',
                'models/gemini-1.5-flash',
                'models/gemini-pro-vision'
            ]
            for pref in preferences:
                if pref in available:
                    print(f"✅ Selected Vision Model: {pref}")
                    return genai.GenerativeModel(pref)
            return genai.GenerativeModel('gemini-pro-vision') # Fallback
        except Exception as e:
            print(f"⚠️ Model selection warning: {e}")
            return genai.GenerativeModel('gemini-pro-vision')

    def extract_text_from_image(self, image_obj: Image.Image) -> Optional[str]:
        """Sends image to Gemini for raw text and LaTeX extraction."""
        prompt = (
            "Transcribe all text and mathematical formulas from this page image accurately. "
            "Convert all math to LaTeX ($...$ for inline, $$...$$ for block). "
            "Do not summarize. Provide verbatim content."
        )
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.vision_model.generate_content([prompt, image_obj])
                return response.text
            except Exception as e:
                print(f"⚠️ Gemini Error (Attempt {attempt+1}): {e}")
                time.sleep(5)
        return None

    def refine_text_to_lecture(self, raw_text: str, context: str) -> Optional[str]:
        """Uses DeepSeek to convert raw OCR text into a Persian lecture note."""
        system_prompt = (
            "You are a Professor at a top university. Rewrite the provided content into a "
            "Comprehensive, Rigorous, and Pedagogical Lecture Note in Persian (Farsi). "
            "- Use Academic Persian tone. "
            "- Keep technical terms in English where necessary. "
            "- Ensure all Math is in correct LaTeX format."
        )
        user_prompt = f"Previous Context: {context[-2000:]}\n\nRaw Content: {raw_text}"

        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.ds_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ DeepSeek Error (Attempt {attempt+1}): {e}")
                time.sleep(3)
        return None

class PDFHandler:
    """Handles PDF file operations and state management."""
    
    @staticmethod
    def get_page_image(doc: fitz.Document, page_num: int) -> Image.Image:
        """Renders a PDF page to a PIL Image."""
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(Config.DPI/72, Config.DPI/72))
        return Image.open(io.BytesIO(pix.tobytes("png")))

    @staticmethod
    def load_state() -> Dict:
        """Loads progress from log file."""
        if Config.LOG_FILE.exists():
            try:
                with open(Config.LOG_FILE, "r", encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {"last_page": 0, "history_context": ""}

    @staticmethod
    def save_state(page_num: int, context: str):
        """Saves current progress."""
        with open(Config.LOG_FILE, "w", encoding='utf-8') as f:
            json.dump({"last_page": page_num, "history_context": context}, f)

class AudioGenerator:
    """Handles Text-to-Speech conversion and Audio processing."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Prepares markdown text for speech synthesis."""
        text = re.sub(r'#+\s', '', text)       # Remove headers
        text = re.sub(r'\*\*|__|\*', '', text) # Remove formatting
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL) # Remove code
        text = text.replace('$', '')           # Remove LaTeX delimiters
        return text

    @staticmethod
    async def generate_audiobook():
        """Converts the final markdown file into a single MP3."""
        print("\n🎧 Generating Audiobook...")
        if not Config.OUTPUT_TEXT.exists():
            print("❌ No text file found to convert.")
            return

        with open(Config.OUTPUT_TEXT, "r", encoding="utf-8") as f:
            full_text = f.read()

        clean_text = AudioGenerator.clean_text(full_text)
        chunks = clean_text.split('\n\n')
        
        temp_files = []
        
        # Generate audio segments
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 5: continue
            
            try:
                temp_path = Config.BASE_DIR / f"temp_{i}.mp3"
                communicate = edge_tts.Communicate(chunk, Config.TTS_VOICE)
                await communicate.save(str(temp_path))
                temp_files.append(temp_path)
                print(".", end="", flush=True)
            except Exception as e:
                print(f"Skipping chunk {i}: {e}")

        # Merge segments
        print("\n💿 Merging Audio Segments...")
        final_audio = AudioSegment.empty()
        for tf in temp_files:
            if tf.exists():
                final_audio += AudioSegment.from_mp3(tf)
                final_audio += AudioSegment.silent(duration=500) # 0.5s pause
                os.remove(tf) 

        final_audio.export(Config.OUTPUT_AUDIO, format="mp3")
        print(f"\n✅ Audiobook Saved: {Config.OUTPUT_AUDIO}")

def main():
    Config.ensure_directories()
    
    # Initialize components
    try:
        ai_processor = AIProcessor()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        return

    # Check PDF
    if not Path(Config.PDF_SOURCE).exists():
        print(f"❌ PDF not found at: {Config.PDF_SOURCE}")
        print("Please update 'PDF_SOURCE' in the Config class.")
        return

    # Load State
    state = PDFHandler.load_state()
    start_page = state["last_page"]
    context = state["history_context"]
    
    doc = fitz.open(Config.PDF_SOURCE)
    total_pages = len(doc)
    print(f"🚀 Starting processing from page {start_page + 1}/{total_pages}")

    # --- Processing Loop ---
    for i in range(start_page, total_pages):
        print(f"\n--- Processing Page {i+1} ---")
        
        # 1. OCR (Vision)
        img = PDFHandler.get_page_image(doc, i)
        raw_text = ai_processor.extract_text_from_image(img)
        
        if not raw_text:
            print("⚠️ Extraction failed, skipping page.")
            continue

        # 2. Refinement (LLM)
        lecture_note = ai_processor.refine_text_to_lecture(raw_text, context)
        
        if lecture_note:
            # Save Content
            with open(Config.OUTPUT_TEXT, "a", encoding="utf-8") as f:
                f.write(f"\n\n# Page {i+1}\n\n{lecture_note}\n\n---\n")
            
            # Update State
            context = lecture_note[-1000:] # Keep last 1000 chars as memory
            PDFHandler.save_state(i + 1, context)
            print(f"✅ Page {i+1} completed.")
        
        time.sleep(2) # Rate limit cooling

    print("\n✨ Processing Complete. Starting Audio Generation...")
    
    # --- Audio Generation ---
    try:
        asyncio.run(AudioGenerator.generate_audiobook())
    except Exception as e:
        print(f"❌ Audio Error: {e}")

if __name__ == "__main__":
    main()