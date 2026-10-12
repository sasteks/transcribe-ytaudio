import os
import re
import subprocess
import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# --- CONFIGURATION ---
YOUTUBE_URL = "https://www.youtube.com/watch?v=Kj2hBspjZOw"
AUDIO_WAV = "broadcast_audio.wav"
TRANSCRIPT_FILE = "broadcast_narrator_english.txt"
EN_SUMMARY_FILE = "structured_vedic_summary_english.txt"
TE_SUMMARY_FILE = "structured_vedic_summary_telugu.txt"
EN_DOCX_FILE = "Sringeri_Navaratri_Vedic_Commentary_English.docx"
TE_DOCX_FILE = "Sringeri_Navaratri_Vedic_Commentary_Telugu.docx"

# Set to True if downloading full video; set to False and specify START/END for a slice
SLICE_ONLY = False
START_TIME = "03:20:00"
END_TIME = "03:35:00"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing pipeline on device: {device.upper()}")

# -------------------------------------------------------------
# STEP 1: Download Audio via yt-dlp
# -------------------------------------------------------------
if not os.path.exists(AUDIO_WAV):
    print("\n[1/4] Downloading audio stream via yt-dlp...")
    dl_cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
        "-o", AUDIO_WAV
    ]
    if SLICE_ONLY:
        dl_cmd.extend(["--download-sections", f"*{START_TIME}-{END_TIME}"])
    dl_cmd.append(YOUTUBE_URL)
    
    subprocess.run(dl_cmd, check=True)
    print(f"Audio ready: {AUDIO_WAV}")
else:
    print(f"\n[1/4] Found existing {AUDIO_WAV}, skipping download.")

# -------------------------------------------------------------
# STEP 2: Whisper Transcription & Translation
# -------------------------------------------------------------
if not os.path.exists(TRANSCRIPT_FILE):
    print("\n[2/4] Transcribing & translating audio with Whisper...")
    whisper_model = whisper.load_model("medium", device=device)
    
    prompt = (
        "Sringeri Sharada Peetham Navaratri puja commentary. Sri Sharadamba, "
        "Hamsavahana Alankara, Brahmi, Suvasini puja, Veda parayana, slokas, "
        "mantras, and philosophical explanation."
    )
    
    result = whisper_model.transcribe(
        AUDIO_WAV,
        language="kn",
        task="translate",
        initial_prompt=prompt,
        temperature=0.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        verbose=True
    )
    
    offset_sec = 0
    if SLICE_ONLY:
        h, m, s = map(int, START_TIME.split(":"))
        offset_sec = h * 3600 + m * 60 + s
        
    filtered_lines = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if re.search(r"(\[Music\]|♪|♫|\[Applause\])", text, re.IGNORECASE) or len(text) == 0:
            continue
        if seg.get("no_speech_prob", 0) > 0.65:
            continue
            
        s_sec = int(seg["start"] + offset_sec)
        e_sec = int(seg["end"] + offset_sec)
        s_ts = f"{s_sec//3600:02d}:{(s_sec%3600)//60:02d}:{s_sec%60:02d}"
        e_ts = f"{e_sec//3600:02d}:{(e_sec%3600)//60:02d}:{e_sec%60:02d}"
        filtered_lines.append(f"[{s_ts} --> {e_ts}] {text}")
        
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(filtered_lines))
    print(f"Transcript written to {TRANSCRIPT_FILE}")
    
    # Free Whisper GPU memory before loading Qwen
    del whisper_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    print(f"\n[2/4] Found existing {TRANSCRIPT_FILE}, skipping Whisper.")

# -------------------------------------------------------------
# STEP 3: LLM Structuring (English & Telugu) via Qwen 2.5
# -------------------------------------------------------------
print("\n[3/4] Structuring Vedic verses and commentary using Qwen 2.5 (3B FP16)...")
with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    clean_entries = [line.strip() for line in f if line.strip() and not line.startswith("=")]

model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

dtype = torch.float16 if device == "cuda" else torch.float32
llm = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None
)

# Chunk entries (~1,800 words per section)
CHUNK_SIZE_WORDS = 1800
chunks, cur_chunk, cur_words = [], [], 0
for entry in clean_entries:
    w = entry.split()
    cur_chunk.append(entry)
    cur_words += len(w)
    if cur_words >= CHUNK_SIZE_WORDS:
        chunks.append("\n".join(cur_chunk))
        cur_chunk, cur_words = [], 0
if cur_chunk:
    chunks.append("\n".join(cur_chunk))

def query_llm(system_msg, user_msg):
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    p_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(p_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=1500, temperature=0.2, do_sample=False)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

en_results, te_results = [], []
for i, chunk_text in enumerate(chunks, start=1):
    print(f" -> Processing Chunk {i}/{len(chunks)}...")
    
    # English
    sys_en = "You are a Vedic scholar. Extract all sacred mantras, philosophical meanings, and ritual details."
    user_en = f"Transcript:\n\"\"\"{chunk_text}\"\"\"\n\nProduce a structured summary strictly listing all Vedic verses/mantras verbatim, their philosophical meaning, and ritual details."
    en_results.append(f"## Broadcast Section {i}\n\n" + query_llm(sys_en, user_en))
    
    # Telugu
    sys_te = "మీరు వైదిక వాఙ్మయం తెలిసిన నిపుణులు. వేద మంత్రాలు, శ్లోకాలు, వాటి అంతరార్థం మరియు పూజా విశేషాలను నిర్దిష్టమైన తెలుగులో వివరించండి."
    user_te = f"వ్యాఖ్యాన అనువాదం:\n\"\"\"{chunk_text}\"\"\"\n\nమంత్రాలను తెలుగు లిపిలో లేదా మూల రూపంలో ఉంచి, తాత్పర్యం మరియు పూజా విధానాన్ని తెలుగు శీర్షికలతో నివేదికగా ఇవ్వండి."
    te_results.append(f"## ప్రసార విభాగం {i}\n\n" + query_llm(sys_te, user_te))

with open(EN_SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write("\n\n---\n\n".join(en_results))

with open(TE_SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write("\n\n---\n\n".join(te_results))

print(f"Generated {EN_SUMMARY_FILE} and {TE_SUMMARY_FILE}")

# -------------------------------------------------------------
# STEP 4: DOCX Export
# -------------------------------------------------------------
print("\n[4/4] Exporting to formatted DOCX documents...")

def export_to_docx(src_txt, dest_docx, title, font_name="Calibri"):
    doc = docx.Document()
    p = doc.add_paragraph()
    r = p.add_run(title)
    r.font.name = font_name
    r.font.size = Pt(20)
    r.bold = True
    r.font.color.rgb = RGBColor(128, 24, 24)
    
    with open(src_txt, "r", encoding="utf-8") as f:
        content = f.read()
        
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        if line.startswith("## "):
            h = doc.add_heading(line.replace("##", "").strip(), level=1)
            for run in h.runs:
                run.font.name = font_name
                run.font.color.rgb = RGBColor(128, 24, 24)
        elif line.startswith("### "):
            h = doc.add_heading(line.replace("###", "").strip(), level=2)
            for run in h.runs:
                run.font.name = font_name
                run.font.color.rgb = RGBColor(60, 40, 20)
        else:
            p = doc.add_paragraph()
            run = p.add_run(line)
            run.font.name = font_name
            run.font.size = Pt(11)
            
    doc.save(dest_docx)
    print(f"Exported: {dest_docx}")

export_to_docx(EN_SUMMARY_FILE, EN_DOCX_FILE, "Sri Sringeri Sharada Peetham - Navaratri Commentary", "Georgia")
export_to_docx(TE_SUMMARY_FILE, TE_DOCX_FILE, "శ్రీ శృంగేరి శారదా పీఠం - నవరాత్రి వ్యాఖ్యానం", "Nirmala UI")

print("\nPipeline execution complete!")