# 🛡️ BorderEye — Border Intelligence & Video Analytics Engine

<p align="center">
  <img src="assets/logo.png" width="220" alt="BorderEye logo" />
</p>

<p align="center">
  <b>The border never sleeps — and neither do we.</b><br/>
  <i>सतर्क सीमा, सुरक्षित देश।</i>
</p>

<p align="center">
  <b>Smart India Hackathon 2026</b> • AI-powered surveillance for border checkpoints and border roads<br/>
  Edge-first • Bandwidth-honest • Tamper-evident • CPU-only
</p>

> *"Every AI-CCTV platform assumes good bandwidth, good cameras, and infinite trust in every alert. Border posts have none of those three."*

---

## ✨ What is BorderEye?

BorderEye is a **software layer over existing CCTV cameras** at border outposts. It watches every frame with AI, raises alerts the instant something happens, and proves every alert is genuine with a tamper-evident ledger. No new cameras. No cloud dependency. No GPU needed.

- 🤖 **YOLOv8 detection + ByteTrack tracking** — persons, vehicles (incl. autorickshaw), persistent IDs
- 🔥 **Virtual fences + pen-drawn tripwires** — draw any line; crossings fire tracked alerts
- 🔢 **ANPR + face detection + night mode** — plates, faces, low-light enhance
- 📡 **Offline-first** — link dies? AI keeps working, events queue, auto-sync on restore
- ⛓️ **Edge Ledger + blockchain anchor** — SHA-256 chained alerts, verifiable, anchorable
- 🆘 **SOS + PDF reports + AI summaries** — command-ready outputs in one click

---

## 🆚 Why BorderEye is different

| Existing CCTV / AI platforms | 🛡️ BorderEye |
|---|---|
| Assume good bandwidth, stream video 24×7 | **Bandwidth-honest** — only ~200 bytes of metadata travel; video never leaves the edge |
| Go blind silently when network drops | **Offline-first** — local queue + auto-sync; the system never goes silent |
| Alerts you must take on faith | **Tamper-evident** — every alert hash-chained; one click verifies the whole log |
| Fixed rectangular zones | **Pen-tool tripwires** — draw any free-form line on the live frame |
| Generic COCO classes | **India-tuned** — autorickshaw class, Hindi action lines, IDD fine-tuning |
| GPU servers, lakh-rupee setups | **CPU-only, tiered hardware** — full post under ~$250, remote node under ~$50 |
| Black-box "AI threat score" | **Documented risk formula** — every point explained, judges can audit it |

---

## 🧠 Tech Stack

| Layer | Technology | Role |
|---|---|---|
| 🖥️ Dashboard | **Streamlit 1.58** | Command-centre UI — feed, alerts, analytics, tripwire canvas |
| ✏️ Drawing | **streamlit-image-coordinates** | Click-to-draw tripwire designer (1:1 with detection frame) |
| 🔍 Detection | **YOLOv8 (Ultralytics 8.4)** | Person + vehicle detection, `detection.pt` fine-tuned on IDD (7 classes) |
| 🎯 Tracking | **ByteTrack (built-in)** | Persistent IDs (`PERSON #3`) via `model.track(persist=True)` |
| 🔢 ANPR | **plate.pt + EasyOCR 1.7** | Plate-region proposals + OCR text |
| 🧑 Faces | **OpenCV Haar cascade 4.14** | Face boxes (classical ML — honest accuracy label in UI) |
| 🌙 Night | **CLAHE (OpenCV)** | Low-light enhance, auto-active 19:00–06:00 |
| 🧠 LLM | **Groq API (`gpt-oss-20b`)** | Alert → 2-line English summary + Hindi `कार्रवाई` line |
| ⛓️ Ledger | **SHA-256 hash chain** | Tamper-evident edge ledger + self-chained anchor log |
| 📄 Reports | **fpdf2** | One-click English PDF incident report (works offline) |
| 🔌 Backend | **FastAPI + WebSocket** | `/api/tracks`, alerts, chain verify — C2 integration contract |
| 🐍 Runtime | **Python 3.14, Torch 2.13 CPU** | Single-language stack, no GPU required |
| 🎨 Theme | **Dark Tactical** (`#0A0D10`) | Inter + JetBrains Mono, sharp 6px geometry |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py dashboard      # → http://localhost:8501
python main.py server         # → API at http://localhost:8000 (/docs for Swagger)
python main.py demo --video path/to/video.mp4
```

**Login** — any email + any 6-digit password (demo gate, e.g. `123456`).
**Groq key (optional)** — put it in `.streamlit/secrets.toml` (never commit it):
```toml
GROQ_API_KEY = "gsk_..."
```

---

## 🖥️ App Workflow

**0. 🔐 Login gate** — full app locked behind demo login; logout wipes the session.

**1. Header** — logo + title, ➕ Add Camera/Footage (RTSP with name/location, or video upload), link/SOS banners.

**2. 📹 Live Detection Feed (4 sources)** — 🎬 Demo Mode • 🎥 Upload Video (4 tabs, play + progress + single-frame OCR inspect) • 📷 Webcam • 🌐 RTSP Stream. Every frame: night-enhance → YOLO + ByteTrack → fence + tripwire checks → face boxes → HUD overlay.

**3. 🚨 Alert Log** — severity, event type, **threat score 0–100** (expandable "why"), timestamp + event ID, ✅ synced / 📤 queued status.

**4. 📊 Dashboard** — metrics, full alert JSON, **low-bandwidth payload preview** (exact bytes on wire), **AI summary**, hash-chain verify.

**5. 📈 Behavioral Analytics** — frames/persons/vehicles/faces, class + hourly charts.

**6. Sidebar** — model status • night/face toggles • simulate buttons • **link toggle + outbox + sync** • **SOS + webhook** • footage/RTSP libraries • **edge ledger + anchor** • **PDF report** • **AI summarizer** • camera health • logout.

**7. ✏️ Tripwire Designer** — sketch any line on a 640×480 canvas → save → magenta TW-lines on live feed → crossings fire tracked alerts.

**8. 📡 Offline flow** — link OFF: AI runs, events queue in `data/outbox.jsonl` → link ON: auto-sync, cards flip to ✅.

---

## 🧪 Tests

```bash
python -m pytest            # pipeline contract tests (~4 s, no weights needed)
```
Plus a headless Streamlit AppTest suite (login gate, SOS, offline queue, anchors, PDF) and a YOLO `track()` smoke test with persistent-ID assertion.

---

## 🗺️ Architecture

```
Camera → YOLO + ByteTrack → Fence/Tripwire/ANPR → Hash-chained Alert
        → Link UP: sync metadata to Command Centre (Streamlit)
        → Link DOWN: local outbox queue → auto-sync on restore
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) • [`docs/ROADMAP.md`](docs/ROADMAP.md) • [`pitch/`](pitch/) • SIH brief: [`docs/BorderEye_SIH_Brief.pdf`](docs/BorderEye_SIH_Brief.pdf)

---

## 💰 Hardware Tiers

| Tier | Hardware | Cost | Runs |
|---|---|---|---|
| Tier 1 — Check post | Jetson Orin Nano | ~$150–250 | Full AI: detect, track, ANPR, fence |
| Tier 2 — Remote node | Microcontroller | ~$20–30 | Motion trigger, store-and-forward |
| Tier 3 — Command | Existing PC/browser | $0 | This dashboard |

---

## 🗣️ Field Doctrines (our slogans)

> *"The border never sleeps — and neither do we."*
> *"Every alert in time is a crisis avoided."*
> *"Vigilance is the price of every peaceful dawn."*
> *"Technology watches where eyes cannot reach."*
> *"सतर्क सीमा, सुरक्षित देश।"*

---

## 📜 License

Smart India Hackathon 2026 — Team BorderEye 🇮🇳
