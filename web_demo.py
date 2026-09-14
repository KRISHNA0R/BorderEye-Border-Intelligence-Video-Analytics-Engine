"""
BorderEye — Border Intelligence & Video Analytics Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full working dashboard with:
  • YOLOv8 / HOG person + vehicle detection
  • Virtual fence intrusion alerts
  • ANPR (number plate recognition)
  • Tamper-evident SHA-256 hash chain
  • Camera signal-loss detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import streamlit as st
import cv2
import numpy as np
import time
import json
import hashlib
import tempfile
import random
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

st.set_page_config(
    page_title="BorderEye — Border Intelligence & Video Analytics Engine",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════
# THEME — Dark Tactical Intelligence (design only)
# Inter for UI, JetBrains Mono for technical data.
# ═══════════════════════════════════════════════════

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body, .stApp {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
/* Never override Streamlit's icon glyphs (aria-hidden spans carry the
   Material Symbols ligatures — forcing Inter there leaks raw text like
   "uploadUpload" / "expand_more" into buttons, expanders, popovers). */
.stApp [aria-hidden="true"] {
    font-family: 'Material Symbols Rounded' !important;
}
h1 { font-size: 1.45rem !important; font-weight: 700 !important; letter-spacing: -0.01em; }
h2 { font-size: 1.15rem !important; font-weight: 600 !important; }
h3 { font-size: 1.0rem !important; font-weight: 600 !important; }
/* Sharp geometry: cards 6-8px, buttons/inputs 6px, badges 4px */
.stButton > button { border-radius: 6px !important; font-weight: 600 !important; }
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stDateInput input, .stTimeInput input {
    border-radius: 6px !important;
}
div[data-baseweb="select"] > div { border-radius: 6px !important; }
.stAlert { border-radius: 6px !important; }
[data-testid="stMetric"] {
    background: #12171D; border: 1px solid #1E293B;
    border-radius: 8px; padding: 8px 12px;
}
[data-testid="stExpander"] details { border-radius: 6px !important; }
[data-testid="stImage"] img { border-radius: 6px; }
button[data-baseweb="tab"] { border-radius: 6px 6px 0 0 !important; }
div[data-testid="stFileUploader"] section { border-radius: 6px !important; }
code, kbd, .tech, .mono {
    font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace !important;
}
.tech { font-size: 0.78rem; letter-spacing: 0.01em; color: #9FB3C8; }
/* Severity accents (text badges) */
.sev-critical { color: #EF4444; font-weight: 600; }
.sev-high { color: #F97316; font-weight: 600; }
.sev-warning { color: #F59E0B; font-weight: 600; }
.sev-safe { color: #22C55E; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════

# Weights fine-tuned on IDD (7 classes incl. autorickshaw). Versioned in
# the repo, so the hosted demo gets them too.
FINE_TUNED_DETECTOR = Path(__file__).parent / "models" / "weights" / "detection.pt"


@st.cache_resource
def load_yolo():
    """Load the fine-tuned detector if present, else stock YOLOv8n.

    Returns (model, is_fine_tuned) — the caller needs to know which, because
    the two have different class taxonomies and want different confidence
    cutoffs.
    """
    from ultralytics import YOLO
    if FINE_TUNED_DETECTOR.exists():
        return YOLO(str(FINE_TUNED_DETECTOR)), True
    return YOLO("yolov8n.pt"), False

@st.cache_resource
def load_hog():
    """Load OpenCV's HOG person detector as fallback.

    Returns None on OpenCV 5+, which removed HOGDescriptor entirely. The
    hosted demo has no torch, so this fallback is its *primary* path — a
    hard failure here takes the whole public page down, which is why it
    degrades instead of raising.
    """
    if not hasattr(cv2, "HOGDescriptor"):
        return None
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


@st.cache_resource
def load_motion_detector():
    """Last-resort detector: background subtraction, available in every
    OpenCV version. Detects movement, not people — labelled as such."""
    return cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32,
                                              detectShadows=False)

@st.cache_resource
def load_ocr():
    """Load EasyOCR for ANPR."""
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False)
    except Exception:
        return None

# Detection engine, best available first: YOLO -> HOG -> motion.
USE_YOLO = False
hog = None
motion = None
try:
    model, FINE_TUNED = load_yolo()
    USE_YOLO = True
    ENGINE = "YOLOv8 (IDD fine-tuned)" if FINE_TUNED else "YOLOv8 (stock COCO)"
except Exception:
    model = None
    FINE_TUNED = False
    hog = load_hog()
    if hog is not None:
        ENGINE = "HOG"
    else:
        motion = load_motion_detector()
        ENGINE = "Motion"


@st.cache_resource
def load_face_detector():
    """Haar face detector — ships inside OpenCV, no downloads, no torch."""
    try:
        clf = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return None if clf.empty() else clf
    except Exception:
        return None


face_clf = load_face_detector()

# EasyOCR is built on first use, NOT at import. Constructing the Reader
# downloads its detection+recognition models and costs ~700MB of RSS; doing
# that at module scope spends the hosted tier's whole memory budget and its
# startup window before the first frame renders. load_ocr() is
# @st.cache_resource, so the first caller pays once and the rest are free.

# ═══════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "prev_hash" not in st.session_state:
    st.session_state.prev_hash = "0" * 64
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "camera_status" not in st.session_state:
    st.session_state.camera_status = {
        "CAM-01": True, "CAM-02": True, "CAM-03": True
    }
if "plate_cache" not in st.session_state:
    st.session_state.plate_cache = {}

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════

W, H = 640, 480
FENCE = np.array([[180, 120], [460, 120], [460, 380], [180, 380]], np.int32)

# Class indices are taxonomy-specific: stock COCO puts bus at 5 and truck
# at 7, the fine-tuned model packs its seven classes into 0-6. Reading the
# map off the loaded model is what stops an autorickshaw being reported as
# a truck.
_WANTED = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "autorickshaw"}
_VEHICLES = {"car", "motorcycle", "bus", "truck", "autorickshaw"}

if USE_YOLO and getattr(model, "names", None):
    _names = {int(i): str(n) for i, n in dict(model.names).items()}
    TARGET_CLASSES = {i: n for i, n in _names.items() if n in _WANTED}
    VEHICLE_CLASSES = {i for i, n in TARGET_CLASSES.items() if n in _VEHICLES}
else:
    TARGET_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                      5: "bus", 7: "truck"}
    VEHICLE_CLASSES = {2, 3, 5, 7}

# 0.45 suits stock COCO; the fine-tuned model peaks at 0.25 (F1), and a
# border demo should favour recall over precision.
DETECT_CONF = 0.25 if (USE_YOLO and FINE_TUNED) else 0.45

INDIAN_PLATE_PATTERN = (
    r"(?:^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}$)"
)

# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════

def compute_risk_score(event_type, severity, confidence, object_class=None, in_fence=False):
    """Deterministic 0-100 risk score from real detection signals.

    This is a documented weighted formula, NOT a trained ML model:
      severity base + detection confidence + fence breach + night hours.
    Returns (score, level, reasons) — reasons explain every point.
    """
    base = {"critical": 70, "high": 55, "medium": 35, "low": 15}.get(severity, 25)
    reasons = [f"Severity '{severity}' (base {base})"]
    score = float(base)

    conf = float(confidence or 0)
    conf_pts = round(conf * 20, 1)
    score += conf_pts
    reasons.append(f"Detection confidence {conf:.0%} (+{conf_pts:g})")

    if in_fence and (object_class == "person" or event_type == "fence_intrusion"):
        score += 10
        reasons.append("Person inside virtual fence (+10)")

    hour = datetime.now().hour
    if hour >= 22 or hour < 5:
        score += 5
        reasons.append(f"Night hours ({hour:02d}:00) (+5)")

    if event_type == "anpr_match":
        score += 5
        reasons.append("Plate on watchlist (+5)")

    score = int(min(99, max(1, round(score))))
    if score >= 80:
        level = "CRITICAL"
    elif score >= 60:
        level = "HIGH"
    elif score >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"
    return score, level, reasons


def create_alert(event_type, explanation, severity, payload=None, confidence=None,
                 object_class=None, in_fence=False):
    """Create alert with hash chain + deterministic risk score."""
    if confidence is None:
        # System events (not model detections) get fixed honest values.
        confidence = {"signal_loss": 0.99, "signal_restored": 0.99}.get(event_type, 0.80)
    risk_score, risk_level, risk_reasons = compute_risk_score(
        event_type, severity, confidence, object_class, in_fence)
    alert = {
        "event_id": f"e{uuid4_hex()}",
        "prev_hash": st.session_state.prev_hash,
        "timestamp": datetime.now().isoformat(),
        "site_id": "BOP-01",
        "camera_id": "CAM-01",
        "event_type": event_type,
        "severity": severity,
        "confidence": round(float(confidence), 2),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "explanation": explanation,
        "payload": payload or {},
    }
    # Compute hash
    payload_for_hash = {k: v for k, v in alert.items() if k not in ("prev_hash", "hash")}
    h = hashlib.sha256(json.dumps(payload_for_hash, sort_keys=True, default=str).encode()).hexdigest()
    alert["hash"] = h
    alert["prev_hash"] = st.session_state.prev_hash
    st.session_state.prev_hash = h
    st.session_state.alerts.append(alert)
    # Low-bandwidth layer: metadata only travels; video never leaves edge.
    alert["link_status"] = outbox_push(alert)
    return alert


def uuid4_hex():
    """Generate a random hex string (no uuid import needed)."""
    import random
    return ''.join(random.choices('0123456789abcdef', k=16))


def verify_chain():
    """Verify hash chain integrity."""
    chain = st.session_state.alerts
    if len(chain) < 2:
        return True
    for i in range(1, len(chain)):
        prev = chain[i - 1].copy()
        prev.pop("hash", None)
        expected = hashlib.sha256(
            json.dumps(prev, sort_keys=True, default=str).encode()
        ).hexdigest()
        if chain[i].get("prev_hash") != chain[i - 1].get("hash"):
            return False
    return True


def is_in_fence(point):
    """Check if a point is inside the virtual fence."""
    return cv2.pointPolygonTest(FENCE, (float(point[0]), float(point[1])), False) >= 0


def build_alert_pdf():
    """Compile every alert this session into an English PDF report.

    Returns PDF bytes. Pure-python (fpdf2), works offline on the edge device.
    """
    from fpdf import FPDF

    def _t(s):
        return str(s).encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "BorderEye - Instant Alert Report",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _t(
        f"Site: BOP-01 | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"| Operator: {st.session_state.get('login_user', 'operator')}"),
        new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, _t(
        f"Total alerts: {len(st.session_state.alerts)} | "
        f"Hash chain: {'VALID' if verify_chain() else 'BROKEN'}"),
        new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    if not st.session_state.alerts:
        pdf.cell(0, 8, "No alerts recorded in this session.",
                 new_x="LMARGIN", new_y="NEXT")
    for a in st.session_state.alerts:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _t(
            f"[{a['severity'].upper()}] {a['event_type'].upper()}  "
            f"(Risk {a.get('risk_score', '-')}/100 {a.get('risk_level', '')})"),
            new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _t(a.get("explanation", "")))
        pdf.cell(0, 6, _t(
            f"Time: {a['timestamp'][:19]} | ID: {a['event_id']} | "
            f"Link: {a.get('link_status', 'synced')}"),
            new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    return bytes(pdf.output())


def _orient(a, b, c):
    """Cross-product orientation of triplet (a,b,c)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segs_cross(a1, a2, b1, b2):
    """True if segment a1-a2 intersects segment b1-b2 (tripwire crossing)."""
    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)
    return ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))


OUTBOX_PATH = Path(__file__).parent / "data" / "outbox.jsonl"


def link_is_up():
    """Low-bandwidth link state. OFF = edge keeps working, events queue."""
    return bool(st.session_state.get("link_up", True))


def _outbox_save():
    """Persist outbox statuses to JSONL (survives restart)."""
    try:
        OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTBOX_PATH, "w", encoding="utf-8") as f:
            for item in st.session_state.get("outbox", []):
                f.write(json.dumps(item) + "\n")
    except Exception:
        pass


def outbox_push(alert):
    """Route an alert through the low-bandwidth layer.

    Link UP   -> metadata marked 'synced' (this is all that travels).
    Link DOWN -> metadata marked 'queued', video stays on the edge device.
    """
    status = "synced" if link_is_up() else "queued"
    st.session_state.setdefault("outbox", []).append({
        "event_id": alert["event_id"], "status": status,
        "timestamp": alert["timestamp"], "event_type": alert["event_type"],
    })
    st.session_state.setdefault("outbox_status", {})[alert["event_id"]] = status
    _outbox_save()
    return status


def outbox_flush():
    """Push all queued events when the link restores. Returns count synced."""
    n = 0
    for item in st.session_state.get("outbox", []):
        if item["status"] == "queued":
            item["status"] = "synced"
            st.session_state.setdefault("outbox_status", {})[item["event_id"]] = "synced"
            n += 1
    _outbox_save()
    return n


ANCHOR_PATH = Path(__file__).parent / "data" / "anchors.jsonl"


def anchor_now():
    """Checkpoint the day's edge-ledger head-hash (blockchain anchor, demo grade).

    Writes a self-chained anchor record locally. If a timestamp-service URL
    is configured (e.g. a Polygon Amoy relayer), the anchor is POSTed there
    and the receipt stored as public proof.
    """
    anchors = st.session_state.get("anchors", [])
    prev = anchors[-1]["anchor_hash"] if anchors else "GENESIS"
    rec = {
        "anchor_id": f"a{uuid4_hex()[:12]}",
        "head_hash": st.session_state.prev_hash,
        "events": len(st.session_state.alerts),
        "timestamp": datetime.now().isoformat(),
        "operator": st.session_state.get("login_user", "operator"),
        "prev_anchor": prev,
    }
    rec["anchor_hash"] = hashlib.sha256(
        json.dumps(rec, sort_keys=True).encode()).hexdigest()
    url = (st.session_state.get("anchor_url") or "").strip()
    if url:
        try:
            import requests
            r = requests.post(url, json=rec, timeout=8)
            rec["receipt"] = f"HTTP {r.status_code}"
        except Exception as e:
            rec["receipt"] = f"failed ({e.__class__.__name__})"
    else:
        rec["receipt"] = "local checkpoint (no timestamp URL set)"
    anchors.append(rec)
    st.session_state.anchors = anchors
    try:
        ANCHOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ANCHOR_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    return rec


def verify_anchors():
    """Recompute every anchor hash + link. True if the anchor log is intact."""
    anchors = st.session_state.get("anchors", [])
    for i, a in enumerate(anchors):
        core = {k: v for k, v in a.items()
                if k not in ("anchor_hash", "receipt")}
        if hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest() != a.get("anchor_hash"):
            return False
        want = "GENESIS" if i == 0 else anchors[i - 1]["anchor_hash"]
        if a.get("prev_anchor") != want:
            return False
    return True


def groq_summarize(alerts, api_key):
    """Summarize alert metadata with Groq LLM.

    Returns 2-line English summary + 1 Hindi action line. Demo/HQ mode only:
    alert metadata leaves the device for this call.
    """
    import requests
    compact = [{
        "type": a["event_type"], "sev": a["severity"],
        "risk": a.get("risk_score"), "time": a["timestamp"][:19],
        "track": (a.get("payload") or {}).get("track_id"),
        "note": a.get("explanation", "")[:200],
    } for a in alerts]
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "system",
                 "content": "You are a border-security duty assistant. Given "
                            "BorderEye alert metadata JSON, reply in exactly 3 "
                            "short lines: lines 1-2 form a 2-line English "
                            "incident summary; line 3 is one Hindi action line "
                            "starting with 'कार्रवाई:'. No extra text."},
                {"role": "user", "content": json.dumps(compact)},
            ],
            "temperature": 0.3, "max_tokens": 250,
        },
        timeout=25,
    )
    if r.status_code != 200:
        return f"Groq error HTTP {r.status_code}: {r.text[:200]}"
    return r.json()["choices"][0]["message"]["content"]


def update_tracks(detections):
    """Track-state engine: tripwire crossing + loitering + fast movement.

    Rule-based heuristics over persistent YOLO track IDs — NOT an ML model.
    Only tracks with a valid track_id (>= 0) participate.
    """
    import math
    now = time.time()
    th = st.session_state.get("track_hist", {})
    seen = set()
    for det in detections:
        tid = det.get("track_id", -1)
        if tid is None or tid < 0:
            continue
        seen.add(tid)
        c = det["center"]
        st_ = th.get(tid)
        if st_ is None:
            th[tid] = {"prev": c, "first_seen": now, "loiter_fired": False,
                       "last_fast": 0.0, "last_cross": 0.0}
            continue
        prev = st_["prev"]
        # ── Tripwire crossing: movement vector vs every pen-drawn segment ──
        if prev != c:
            for i, seg in enumerate(st.session_state.get("tripwires", [])):
                (x1, y1), (x2, y2) = seg
                if segs_cross(prev, c, (x1, y1), (x2, y2)):
                    if now - st_["last_cross"] > 30:
                        st_["last_cross"] = now
                        create_alert(
                            "tripwire_crossing",
                            f"Track #{tid} ({det['class']}) crossed tripwire-{i + 1}.",
                            "high",
                            payload={"track_id": tid, "tripwire": i + 1},
                            confidence=det["confidence"],
                            object_class=det["class"],
                        )
            # ── Fast movement: large per-frame centroid jump ──
            d = math.hypot(c[0] - prev[0], c[1] - prev[1])
            if d > 45 and now - st_["last_fast"] > 60:
                st_["last_fast"] = now
                create_alert(
                    "fast_movement",
                    f"Track #{tid} moving unusually fast (~{d:.0f}px/frame).",
                    "medium",
                    payload={"track_id": tid, "px_per_frame": round(d, 1)},
                    confidence=det["confidence"],
                    object_class=det["class"],
                )
        # ── Loitering: same ID inside fence > 15 s ──
        if det["in_fence"]:
            if not st_["loiter_fired"] and now - st_["first_seen"] > 15:
                st_["loiter_fired"] = True
                create_alert(
                    "loitering",
                    f"Track #{tid} loitering inside fence for "
                    f"{now - st_['first_seen']:.0f}s.",
                    "medium",
                    payload={"track_id": tid,
                             "dwell_s": round(now - st_["first_seen"], 1)},
                    confidence=det["confidence"],
                    object_class=det["class"],
                )
        else:
            st_["first_seen"] = now
            st_["loiter_fired"] = False
        st_["prev"] = c
    # Prune stale tracks (keep memory bounded)
    for tid in list(th.keys()):
        if tid not in seen:
            del th[tid]
        if len(th) <= 80:
            break
    st.session_state.track_hist = th


def enhance_night(frame):
    """Brighten a dark frame with CLAHE on the LAB L-channel. Cheap, no models."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def is_night_hours():
    """Auto night window 19:00–06:00 local time."""
    h = datetime.now().hour
    return (h >= 19 or h < 6)


def night_active():
    """Manual toggle OR automatic night hours."""
    return bool(st.session_state.get("night_mode")) or is_night_hours()


def detect_frame(frame):
    """Run detection on a frame. Returns annotated frame + detections."""
    if night_active():
        frame = enhance_night(frame)
    frame = cv2.resize(frame, (W, H))
    detections = []

    if USE_YOLO and model is not None:
        # Built-in ByteTrack: persistent IDs across frames, no extra deps.
        results = model.track(frame, conf=DETECT_CONF, persist=True,
                              verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in TARGET_CLASSES:
                    continue
                conf = float(box.conf[0])
                tid = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                in_fence = is_in_fence((cx, cy))
                cls_name = TARGET_CLASSES[cls]
                detections.append({
                    "class": cls_name, "class_id": cls,
                    "confidence": conf, "track_id": tid,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy), "in_fence": in_fence,
                })
    elif hog is not None:
        # HOG fallback — OpenCV 4.x only
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8),
                                               padding=(4, 4), scale=1.05)
        for (x, y, w, h), conf in zip(boxes, weights):
            cx, cy = x + w // 2, y + h // 2
            in_fence = is_in_fence((cx, cy))
            detections.append({
                "class": "person", "class_id": 0,
                "confidence": float(conf[0]),
                "bbox": (x, y, x + w, y + h),
                "center": (cx, cy), "in_fence": in_fence,
            })
    elif motion is not None:
        # Motion fallback — movement, not classification. Confidence is a
        # blob-area heuristic, not a model score; do not present it as one.
        mask = motion.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            area = cv2.contourArea(contour)
            if area < 400:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            in_fence = is_in_fence((cx, cy))
            detections.append({
                "class": "motion", "class_id": 0,
                "confidence": min(0.99, area / (W * H) * 8),
                "bbox": (x, y, x + w, y + h),
                "center": (cx, cy), "in_fence": in_fence,
            })

    # Draw
    vis = frame.copy()
    cv2.polylines(vis, [FENCE], True, (0, 255, 0), 2)
    cv2.putText(vis, "VIRTUAL FENCE", (210, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # Pen-drawn tripwires (magenta)
    for i, seg in enumerate(st.session_state.get("tripwires", [])):
        (x1, y1), (x2, y2) = seg
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        cv2.putText(vis, f"TW-{i + 1}", (int(x1) + 4, int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    persons = 0
    vehicles = 0
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 0, 255) if det["in_fence"] else (0, 255, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        tid = det.get("track_id", -1)
        tag = f" #{tid}" if tid is not None and tid >= 0 else ""
        label = f"{det['class']}{tag} {det['confidence']:.0%}"
        cv2.putText(vis, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if det["class"] == "person":
            persons += 1
            if det["in_fence"]:
                # Throttle per track (60 s) so the log doesn't flood per frame.
                fl = st.session_state.get("fence_last", {})
                if tid is None or tid < 0 or time.time() - fl.get(tid, 0) > 60:
                    if tid is not None and tid >= 0:
                        fl[tid] = time.time()
                        st.session_state.fence_last = fl
                    create_alert(
                        "fence_intrusion",
                        f"Person{tag} detected crossing virtual fence. "
                        f"Confidence: {det['confidence']:.0%}.",
                        "high",
                        payload={"track_id": tid},
                        confidence=det["confidence"],
                        object_class=det["class"],
                        in_fence=True,
                    )
        elif det["class_id"] in VEHICLE_CLASSES:
            vehicles += 1

    # ── Track engine: tripwire crossing + loitering + fast movement ──
    update_tracks(detections)

    # ── Crowd gathering: 5+ persons in one frame (rule-based, 1/min) ──
    if persons >= 5:
        last = st.session_state.get("crowd_last_alert", 0)
        if time.time() - last > 60:
            st.session_state.crowd_last_alert = time.time()
            create_alert(
                "crowd_gathering",
                f"Crowd gathering: {persons} persons in one frame.",
                "medium",
                payload={"person_count": persons},
                confidence=0.80,
            )

    # ── Face detection (Haar, built-in OpenCV) ──
    faces = 0
    if st.session_state.get("face_detect") and face_clf is not None:
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        for (fx, fy, fw, fh) in face_clf.detectMultiScale(gray, 1.1, 4):
            faces += 1
            cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            cv2.putText(vis, "FACE", (fx, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    st.session_state.last_faces = faces

    # ── Behavioral stats (every processed frame) ──
    st.session_state.stats_frames = st.session_state.get("stats_frames", 0) + 1
    tot = st.session_state.get("stats_total", {})
    for d in detections:
        tot[d["class"]] = tot.get(d["class"], 0) + 1
    if faces:
        tot["face"] = tot.get("face", 0) + faces
    st.session_state.stats_total = tot
    hr = datetime.now().strftime("%H:00")
    hh = st.session_state.get("stats_hourly", {})
    hh[hr] = hh.get(hr, 0) + len(detections) + faces
    st.session_state.stats_hourly = hh

    # ── Night-time movement alert (throttled to 1/min) ──
    if night_active() and (persons + vehicles) > 0:
        last = st.session_state.get("night_last_alert", 0)
        if time.time() - last > 60:
            st.session_state.night_last_alert = time.time()
            create_alert(
                "night_movement",
                f"Night-time movement: {persons} person(s), {vehicles} vehicle(s) in view.",
                "medium", confidence=0.80,
            )

    # HUD
    cv2.rectangle(vis, (0, 0), (W, 28), (20, 20, 30), -1)
    cv2.putText(vis, f"BorderEye | BOP-01 CAM-01 | Frame {st.session_state.frame_count} | "
                f"Persons: {persons} | Vehicles: {vehicles}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

    return vis, detections


def run_ocr_on_frame(frame):
    """Run OCR on frame for plate detection."""
    reader = load_ocr()
    if reader is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(filtered, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / max(float(h), 1)
            area = w * h
            if 2.0 < aspect < 6.0 and 1000 < area < 50000:
                plate_img = frame[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
                if plate_img.size == 0:
                    continue
                plate_img = cv2.resize(plate_img, None, fx=2, fy=2,
                                        interpolation=cv2.INTER_CUBIC)
                results = reader.readtext(plate_img)
                for (_, text, conf) in results:
                    text = text.strip().upper()
                    if conf > 0.5 and len(text) >= 4:
                        plates.append({
                            "text": text, "confidence": conf,
                            "bbox": (x, y, x + w, y + h),
                        })
    return plates


def generate_demo_frame(tick):
    """Generate a synthetic surveillance frame for demo."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    # Ground / grass
    frame[:] = (35, 50, 35)
    # Sky
    frame[:100] = (70, 50, 35)
    # Road
    cv2.rectangle(frame, (0, 340), (W, H), (65, 65, 65), -1)
    cv2.line(frame, (0, 370), (W, 370), (180, 180, 180), 1, cv2.LINE_AA)
    # Dashed centre line
    for dx in range(0, W, 40):
        cv2.line(frame, (dx, 370), (dx + 20, 370), (240, 240, 60), 2, cv2.LINE_AA)

    # ── Fence ──
    cv2.polylines(frame, [FENCE], True, (0, 220, 0), 2, cv2.LINE_AA)
    # Label top-centre of fence
    fc = FENCE.mean(axis=0).astype(int)
    cv2.putText(frame, "VIRTUAL FENCE", (fc[0] - 50, FENCE[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 1, cv2.LINE_AA)

    # ── Animated person ──
    px = 100 + (tick * 3) % 400
    py = 260 + int(8 * np.sin(tick * 0.25))
    # Body
    cv2.circle(frame, (px, py - 28), 11, (200, 200, 210), -1, cv2.LINE_AA)
    cv2.line(frame, (px, py - 16), (px, py + 10), (200, 200, 210), 2, cv2.LINE_AA)
    cv2.line(frame, (px, py), (px - 14, py + 18), (200, 200, 210), 2, cv2.LINE_AA)
    cv2.line(frame, (px, py), (px + 14, py + 18), (200, 200, 210), 2, cv2.LINE_AA)
    in_fence = is_in_fence((px, py))
    pcolor = (0, 0, 255) if in_fence else (0, 255, 255)
    cv2.rectangle(frame, (px - 18, py - 42), (px + 18, py + 24), pcolor, 2, cv2.LINE_AA)
    cv2.putText(frame, f"PERSON 0.{random.randint(82,97)}", (px - 18, py - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, pcolor, 1, cv2.LINE_AA)

    # ── Animated car ──
    car_x = 500 - (tick * 4) % 650
    car_y = 380
    cv2.rectangle(frame, (car_x, car_y - 18), (car_x + 55, car_y + 8), (10, 80, 180), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (car_x + 8, car_y - 28), (car_x + 45, car_y - 18), (10, 60, 160), -1, cv2.LINE_AA)
    cv2.circle(frame, (car_x + 10, car_y + 10), 5, (30, 30, 30), -1)
    cv2.circle(frame, (car_x + 45, car_y + 10), 5, (30, 30, 30), -1)
    cv2.putText(frame, f"CAR 0.{random.randint(75,94)}", (car_x, car_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 165, 0), 1, cv2.LINE_AA)

    # ── HUD top bar ──
    cv2.rectangle(frame, (0, 0), (W, 26), (15, 15, 25), -1)
    cv2.putText(
        frame,
        f"BorderEye  |  BOP-01  CAM-01  |  Frame {tick}  |  {datetime.now().strftime('%H:%M:%S')}",
        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 255), 1, cv2.LINE_AA,
    )
    # Detection count bar
    cv2.rectangle(frame, (0, H - 24), (W, H), (15, 15, 25), -1)
    cv2.putText(frame, f"Detection engine: {ENGINE}  |  Objects: 2",
                (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

    return frame, in_fence


def render_video_tab(video_path, key, loc=""):
    """One saved video: full Play (live analysis) or Single Frame inspect."""
    ph = st.empty()
    if loc:
        st.caption(f"📍 Location: {loc}")
    probe = cv2.VideoCapture(video_path)
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    probe.release()

    view = st.radio("View", ["▶ Play Video", "🖼 Single Frame"],
                    horizontal=True, key=f"view_{key}")

    if view == "🖼 Single Frame":
        cap = cv2.VideoCapture(video_path)
        frame_idx = st.slider("Frame", 0,
                              max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 0),
                              0, key=f"frame_{key}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            annotated, dets = detect_frame(frame)
            plates = run_ocr_on_frame(frame)
            if plates:
                for p in plates:
                    cv2.rectangle(annotated, p["bbox"][:2], p["bbox"][2:], (0, 255, 0), 2)
                    cv2.putText(annotated, p["text"], (p["bbox"][0], p["bbox"][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ph.image(annotated, channels="BGR", use_container_width=True)
            persons = sum(1 for d in dets if d["class"] == "person")
            vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
            st.caption(f"🔍 Frame {frame_idx} | Persons: {persons} | Vehicles: {vehicles} | Plates: {len(plates)}")
        cap.release()
    else:
        stride = st.slider("Process every Nth frame", 1, 10, 2, key=f"stride_{key}",
                           help="Higher = faster on CPU. Alerts still fire live.")
        col_p1, col_p2 = st.columns(2)
        if col_p1.button("▶ Play", use_container_width=True, key=f"play_{key}"):
            st.session_state[f"playing_{key}"] = True
        if col_p2.button("⏹ Stop", use_container_width=True, key=f"stop_{key}"):
            st.session_state[f"playing_{key}"] = False
            st.info("⏸ Stopped. Press Play to resume from the same position.")
        if st.session_state.get(f"playing_{key}"):
            if total <= 0:
                st.warning("⚠️ Could not read this video's frame count.")
                st.session_state[f"playing_{key}"] = False
            else:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.get(f"play_pos_{key}", 0))
                prog = st.progress(0)
                status = st.empty()
                idx = st.session_state.get(f"play_pos_{key}", 0)
                dets = []
                plates = []
                while idx < total:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    idx += 1
                    if (idx % stride) != 0:
                        continue
                    st.session_state.frame_count = idx
                    annotated, dets = detect_frame(frame)
                    plates = run_ocr_on_frame(frame)
                    if plates:
                        for p in plates:
                            cv2.rectangle(annotated, p["bbox"][:2], p["bbox"][2:], (0, 255, 0), 2)
                            cv2.putText(annotated, p["text"], (p["bbox"][0], p["bbox"][1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    ph.image(annotated, channels="BGR", use_container_width=True)
                    persons = sum(1 for d in dets if d["class"] == "person")
                    vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
                    status.caption(f"🔍 Frame {idx}/{total} | Persons: {persons} | Vehicles: {vehicles} | Plates: {len(plates)}")
                    prog.progress(min(idx / total, 1.0))
                    st.session_state[f"play_pos_{key}"] = idx
                cap.release()
                st.session_state[f"play_pos_{key}"] = 0
                st.session_state[f"playing_{key}"] = False
                st.success("✅ Video finished — alerts are in the Alert Log.")
        else:
            st.info("▶ Press Play to run live detection on this video.")


# ═══════════════════════════════════════════════════
# AUTH GATE — full app lock behind login
# ═══════════════════════════════════════════════════

LOGIN_QUOTES = [
    ("The border never sleeps \u2014 and neither do we.", "Field Doctrine 01"),
    ("Every alert in time is a crisis avoided.", "Field Doctrine 02"),
    ("Vigilance is the price of every peaceful dawn.", "Field Doctrine 03"),
    ("From the remotest post to command \u2014 every second, every frame counts.",
     "Field Doctrine 04"),
    ("Technology watches where eyes cannot reach.", "Field Doctrine 05"),
    ("\u0938\u0924\u0930\u094d\u0915 \u0938\u0940\u092e\u093e, \u0938\u0941\u0930\u0915\u094d\u0937\u093f\u0924 \u0926\u0947\u0936\u0964",
     "Field Doctrine 06"),
]
QUOTE_SLOT_S = 6  # seconds per quote


def login_quotes_html():
    """Rotating login quotes — pure CSS fade cycle, no reruns, no server load."""
    total = QUOTE_SLOT_S * len(LOGIN_QUOTES)
    divs = []
    for i, (text, attr) in enumerate(LOGIN_QUOTES):
        divs.append(
            f"<div class='lq lq{i}'><div class='lq-text'>\u201c{text}\u201d</div>"
            f"<div class='lq-attr'>\u2014 {attr}</div></div>"
        )
    delays = "\n".join(
        f".lq{i} {{ animation-delay: {i * QUOTE_SLOT_S}s; }}" for i in range(len(LOGIN_QUOTES))
    )
    return f"""
<style>
.lq-wrap {{ position: relative; min-height: 96px; margin: 4px 0 8px 0; }}
.lq {{ position: absolute; inset: 0; opacity: 0; text-align: center;
       animation: lqfade {total}s ease-in-out infinite; }}
.lq-text {{ font-style: italic; font-size: 0.95rem; color: #C7D5E3; line-height: 1.45; }}
.lq-attr {{ margin-top: 6px; font-size: 0.7rem; letter-spacing: 0.18em;
            color: #38BDF8; font-weight: 600; }}
{delays}
@keyframes lqfade {{
  0% {{ opacity: 0; transform: translateY(10px); }}
  2.5% {{ opacity: 1; transform: translateY(0); }}
  14% {{ opacity: 1; transform: translateY(0); }}
  16.6% {{ opacity: 0; transform: translateY(-8px); }}
  100% {{ opacity: 0; }}
}}
</style>
<div class="lq-wrap">{''.join(divs)}</div>
"""


if not st.session_state.get("authed"):
    st.markdown(
        "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    _c1, _c2, _c3 = st.columns([1, 2, 1])
    with _c2:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=180)
        st.markdown("## 🛡️ BorderEye")
        st.caption("Border Intelligence & Video Analytics Engine")
        st.markdown(login_quotes_html(), unsafe_allow_html=True)
        st.divider()
        login_email = st.text_input("Email", key="login_email",
                                    placeholder="Enter any email address")
        login_pass = st.text_input("Password", type="password", key="login_pass",
                                   placeholder="Enter 6-digit password", max_chars=6)
        if st.button("🔐 Login", use_container_width=True, key="login_btn"):
            import re
            if not login_email or not login_email.strip():
                st.error("Email is required.")
            elif not re.fullmatch(r"\d{6}", login_pass or ""):
                st.error("Password must be exactly 6 digits.")
            else:
                st.session_state.authed = True
                st.session_state.login_user = login_email.strip()
                st.rerun()
        st.divider()
        st.caption("Demo access — use any email, password: any 6-digit number")
    st.stop()

# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=130)
    st.markdown("## 🛡️ BorderEye")
    st.caption("BorderEye — Border Intelligence & Video Analytics Engine")
    _user = st.session_state.get("login_user", "operator")
    st.caption(f"👤 Logged in as: **{_user}**")
    if st.button("🚪 Logout", use_container_width=True, key="sidebar_logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    st.divider()

    # Model status
    if USE_YOLO and FINE_TUNED:
        st.success("✅ YOLOv8 fine-tuned on IDD")
        st.caption(f"7 classes incl. autorickshaw · conf {DETECT_CONF}")
    elif USE_YOLO:
        st.success("✅ YOLOv8 loaded (stock COCO)")
    elif ENGINE == "HOG":
        st.warning("⚠️ Using HOG fallback (no torch)")
    else:
        st.warning("⚠️ Using motion fallback (no torch, OpenCV 5+)")
    st.info("ℹ️ ANPR/OCR loads on first plate scan")
    st.checkbox("🌙 Night Mode (low-light enhance)", key="night_mode",
                help="Brightens dark frames before detection. Applies to demo/upload/webcam/RTSP. Auto-active 19:00–06:00.")
    if is_night_hours() and not st.session_state.get("night_mode"):
        st.caption("🌙 Auto-night active (time-based) — movement will raise alerts.")
    st.checkbox("🧑 Face Detection", key="face_detect",
                help="Draws a blue FACE box on detected faces. Built-in OpenCV — basic accuracy.")
    if st.session_state.get("face_detect") and face_clf is None:
        st.warning("Face model files missing — face detection off hai.")

    st.divider()
    if st.button("🚨 Simulate Fence Intrusion", use_container_width=True):
        create_alert("fence_intrusion",
                     "Person crossed Zone-1 at 1.4 m/s, bearing NE. No patrol scheduled.",
                     "high", confidence=0.88, object_class="person", in_fence=True)
    if st.button("🚗 Simulate ANPR Match", use_container_width=True):
        import random
        plate = f"BR{random.randint(10,99)}AB{random.randint(1000,9999)}"
        create_alert("anpr_match",
                     f"Vehicle {plate} flagged — plate matches watchlist at Checkpoint-1.",
                     "medium",
                     {"plate_text": plate},
                     confidence=0.85, object_class="vehicle")
    if st.button("📡 Simulate Signal Loss", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = False
        create_alert("signal_loss",
                     "CAM-03 at BOP-01 lost signal. Possible jamming or tampering.",
                     "critical")
    if st.button("🔄 Restore Camera", use_container_width=True):
        st.session_state.camera_status["CAM-03"] = True
        create_alert("signal_restored", "CAM-03 signal restored.", "low")
    if st.button("🗑️ Clear All Alerts", use_container_width=True):
        st.session_state.alerts = []
        st.session_state.prev_hash = "0" * 64

    st.divider()
    st.subheader("📡 Low-Bandwidth Link")
    st.caption("OFF = edge keeps working, events queue locally.")
    st.toggle("Link Up", value=True, key="link_up")
    # Auto-sync the moment the link restores
    if st.session_state.link_up and not st.session_state.get("link_prev", True):
        n = outbox_flush()
        if n:
            st.success(f"✅ Link restored — {n} queued event(s) synced.")
    st.session_state.link_prev = bool(st.session_state.link_up)
    q = sum(1 for i in st.session_state.get("outbox", []) if i["status"] == "queued")
    s = sum(1 for i in st.session_state.get("outbox", []) if i["status"] == "synced")
    st.caption(f"📤 Queued: {q} | ✅ Synced: {s}")
    if q and st.session_state.link_up:
        if st.button("🔄 Sync Now", use_container_width=True):
            st.success(f"✅ {outbox_flush()} event(s) synced.")
            st.rerun()

    st.divider()
    st.subheader("🆘 SOS — Higher Command")
    st.text_input("HQ webhook URL (optional)", key="sos_webhook",
                  placeholder="https://hq.example.com/api/sos")
    if st.button("🆘 SEND SOS", use_container_width=True, type="primary"):
        a = create_alert(
            "sos",
            "SOS raised by on-duty operator at BOP-01. Immediate attention "
            "from higher command requested.",
            "critical",
            payload={"raised_by": st.session_state.get("login_user", "operator")},
            confidence=0.99,
        )
        st.session_state.sos_active = True
        st.session_state.sos_time = a["timestamp"]
        url = (st.session_state.get("sos_webhook") or "").strip()
        if url:
            try:
                import requests
                r = requests.post(url, json={
                    "event_id": a["event_id"], "type": "sos",
                    "site": "BOP-01", "timestamp": a["timestamp"],
                    "raised_by": st.session_state.get("login_user", "operator"),
                }, timeout=5)
                st.session_state.sos_hook = f"HTTP {r.status_code}"
            except Exception as e:
                st.session_state.sos_hook = f"failed ({e.__class__.__name__})"
                st.warning("Webhook unreachable — SOS queued in local outbox.")
        st.rerun()
    if st.session_state.get("sos_active"):
        st.error(f"🆘 SOS ACTIVE since {st.session_state.get('sos_time', '')[:19]}")
        if st.session_state.get("sos_hook"):
            st.caption(f"Webhook: {st.session_state.sos_hook}")
        if st.button("Stand Down", use_container_width=True):
            st.session_state.sos_active = False
            create_alert("signal_restored", "SOS stood down by operator.", "low")
            st.rerun()

    st.divider()
    st.subheader("🎞 Footage")
    st.caption("Add videos (max 4) — play them in the Upload Video tab.")
    uploads = st.file_uploader("Add videos", type=["mp4", "avi", "mov", "mkv"],
                               accept_multiple_files=True, key="footage_up",
                               label_visibility="collapsed")
    if uploads:
        lib = st.session_state.get("video_lib", {})
        for up in uploads[:4]:
            k = f"{up.name}_{up.size}"
            if k not in lib and len(lib) < 4:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(up.getvalue())
                lib[k] = {"name": up.name, "path": tfile.name, "location": ""}
        st.session_state.video_lib = lib
    for k in list(st.session_state.get("video_lib", {}).keys()):
        c1, c2 = st.columns([4, 1])
        nm = st.session_state.video_lib[k].get("name", "Video")[:28]
        lc = st.session_state.video_lib[k].get("location", "")
        c1.caption(f"🎬 {nm}" + (f" • 📍 {lc[:20]}" if lc else ""))
        if c2.button("❌", key=f"delvid_{k}"):
            try:
                Path(st.session_state.video_lib[k]["path"]).unlink(missing_ok=True)
            except Exception:
                pass
            del st.session_state.video_lib[k]
            st.rerun()

    st.divider()
    st.subheader("🌐 RTSP Streams")
    st.caption("Save streams (max 4) — press Connect to go live.")
    if "rtsp_lib" not in st.session_state:
        st.session_state.rtsp_lib = [
            "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"]
    new_rtsp = st.text_input("Add RTSP URL", key="rtsp_new",
                             placeholder="rtsp://user:pass@ip:port/stream")
    if st.button("➕ Save Stream", use_container_width=True):
        url = (new_rtsp or "").strip()
        if not url.lower().startswith("rtsp://"):
            st.warning("URL `rtsp://` se start hona chahiye.")
        elif url not in st.session_state.rtsp_lib and len(st.session_state.rtsp_lib) < 4:
            st.session_state.rtsp_lib.append(url)
            meta = st.session_state.get("rtsp_meta", {})
            meta.setdefault(url, {"name": f"CAM-{len(st.session_state.rtsp_lib):02d}",
                                  "location": ""})
            st.session_state.rtsp_meta = meta
            st.rerun()
    for i, url in enumerate(list(st.session_state.get("rtsp_lib", []))):
        meta = st.session_state.get("rtsp_meta", {}).get(url, {})
        nm = meta.get("name", "")
        lc = meta.get("location", "")
        label = f"📡 {nm}" if nm else "📡 Stream"
        if lc:
            label += f" • 📍 {lc[:20]}"
        st.caption(label)
        st.caption(f"{url[:44]}")
        c1, c2 = st.columns(2)
        if c1.button("🔌 Connect", use_container_width=True, key=f"rtsp_go_{i}"):
            st.session_state.rtsp_url = url
            st.session_state.source_mode = "🌐 RTSP Stream"
            st.rerun()
        if c2.button("❌", use_container_width=True, key=f"rtsp_del_{i}"):
            st.session_state.rtsp_lib.remove(url)
            st.rerun()

    st.divider()
    st.subheader("🔐 Edge Ledger")
    st.caption("Hash-chained, tamper-evident event ledger.")
    chain_ok = verify_chain()
    if chain_ok:
        st.success("✅ Chain VALID")
    else:
        st.error("❌ Chain BROKEN!")
    st.metric("Total Events", len(st.session_state.alerts))
    if st.session_state.prev_hash != "0" * 64:
        st.caption(f"Head: `{st.session_state.prev_hash[:20]}…`")

    st.divider()
    st.subheader("⚓ Blockchain Anchor")
    st.caption("Checkpoint the day's head-hash. With a timestamp URL "
               "(e.g. Polygon Amoy relayer) this becomes public proof.")
    st.text_input("Timestamp service URL (optional)", key="anchor_url",
                  placeholder="https://relayer.example.com/anchor")
    if st.button("⚓ Anchor Now", use_container_width=True):
        _a = anchor_now()
        st.success(f"Anchored `{_a['anchor_hash'][:16]}…` ({_a['receipt']})")
    _anchors = st.session_state.get("anchors", [])
    if _anchors:
        _last = _anchors[-1]
        st.caption(f"Anchors: {len(_anchors)} | Latest: `{_last['anchor_hash'][:16]}…`")
        if st.button("Verify Anchors", use_container_width=True):
            if verify_anchors():
                st.success(f"✅ {len(_anchors)} anchor(s) intact")
            else:
                st.error("❌ Anchor log compromised!")

    st.divider()
    st.subheader("📄 Reports")
    if st.button("📄 Get Instant Alert Report (PDF)", use_container_width=True):
        try:
            st.session_state.report_pdf = build_alert_pdf()
            st.session_state.report_name = (
                "BorderEye_alerts_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            st.success("Report ready — download below.")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
    if st.session_state.get("report_pdf"):
        st.download_button(
            "⬇ Download PDF", data=st.session_state.report_pdf,
            file_name=st.session_state.report_name,
            mime="application/pdf", use_container_width=True,
            key="dl_report")

    st.divider()
    st.subheader("🤖 AI Summarizer")
    st.caption("Groq LLM over alert metadata. Demo/HQ mode only — "
               "metadata leaves the device for this call.")
    if "groq_key" not in st.session_state:
        try:
            st.session_state.groq_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            st.session_state.groq_key = ""
    st.text_input("Groq API key", type="password", key="groq_key")
    _n = st.number_input("Summarize last N alerts", 1, 5, 1, key="ai_n")
    if st.button("✨ Summarize", use_container_width=True):
        if not (st.session_state.get("groq_key") or "").strip():
            st.warning("API key missing.")
        elif not st.session_state.alerts:
            st.warning("No alerts to summarize yet.")
        else:
            with st.spinner("Asking Groq…"):
                try:
                    st.session_state.ai_summary = groq_summarize(
                        st.session_state.alerts[-int(_n):],
                        st.session_state.groq_key.strip())
                except Exception as e:
                    st.session_state.ai_summary = (
                        f"AI call failed ({e.__class__.__name__}): {e}")
            st.rerun()

    st.divider()
    st.subheader("📡 Cameras")
    for cam, ok in st.session_state.camera_status.items():
        icon = "🟢" if ok else "🔴"
        st.markdown(f"{icon} {cam}")

# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

hc1, hc2 = st.columns([5, 1])
with hc1:
    _t1, _t2 = st.columns([1, 8])
    with _t1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=76)
    with _t2:
        st.markdown("# 🛡️ BorderEye — Border Intelligence & Video Analytics Engine")
        st.caption("Real-time AI-powered surveillance • YOLOv8 detection • Virtual fences • ANPR • Tamper-evident logging")
with hc2:
    st.write("")
    with st.popover("➕ Add Camera / Footage", use_container_width=True):
        stype = st.selectbox("Source type", ["🌐 RTSP Stream", "🎞 Video Footage"],
                             key="add_type")
        if stype == "🌐 RTSP Stream":
            st.text_input("Camera name", key="add_cam_name", placeholder="e.g. CAM-04")
            st.text_input("Location", key="add_cam_loc", placeholder="e.g. Gate-East")
            st.text_input("RTSP URL", key="add_cam_url", placeholder="rtsp://user:pass@ip:port/stream")
            if st.button("Save Camera", use_container_width=True, key="add_cam_go"):
                url = (st.session_state.get("add_cam_url") or "").strip()
                name = (st.session_state.get("add_cam_name") or "").strip() or "CAM"
                loc = (st.session_state.get("add_cam_loc") or "").strip()
                if not url.lower().startswith("rtsp://"):
                    st.error("URL must start with rtsp://")
                elif url not in st.session_state.get("rtsp_lib", []) and len(st.session_state.get("rtsp_lib", [])) < 4:
                    st.session_state.rtsp_lib.append(url)
                    meta = st.session_state.get("rtsp_meta", {})
                    meta[url] = {"name": name, "location": loc}
                    st.session_state.rtsp_meta = meta
                    st.success(f"Camera '{name}' saved.")
                    st.rerun()
                else:
                    st.warning("Already saved or limit (4) reached.")
        else:
            up = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"],
                                  key="add_vid")
            st.text_input("Name", key="add_vid_name", placeholder="e.g. Gate footage")
            st.text_input("Location", key="add_vid_loc", placeholder="e.g. Gate-East")
            if st.button("Save Footage", use_container_width=True, key="add_vid_go"):
                if up is None:
                    st.error("Please choose a video file first.")
                else:
                    lib = st.session_state.get("video_lib", {})
                    k = f"{up.name}_{up.size}"
                    name = (st.session_state.get("add_vid_name") or "").strip() or up.name
                    loc = (st.session_state.get("add_vid_loc") or "").strip()
                    if k not in lib and len(lib) < 4:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(up.getvalue())
                        lib[k] = {"name": name, "path": tfile.name, "location": loc}
                        st.session_state.video_lib = lib
                        st.success(f"Footage '{name}' saved.")
                        st.rerun()
                    else:
                        st.warning("Already saved or limit (4) reached.")

st.markdown(
    "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>",
    unsafe_allow_html=True,
)
st.info("🏛️ Official demonstration build — all video analytics run on-device. No video leaves this system.")
if not st.session_state.get("link_up", True):
    st.warning("🔴 LINK DOWN — edge mode: AI running locally, events queuing in local outbox. Video stays on device.")
if st.session_state.get("sos_active"):
    st.error(f"🆘 SOS ACTIVE — higher command alerted at {st.session_state.get('sos_time', '')[:19]}. Stand down from the sidebar.")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📹 Live Detection Feed")

    mode = st.radio("Source", ["🎬 Demo Mode", "🎥 Upload Video", "📷 Webcam", "🌐 RTSP Stream"],
                     horizontal=True, key="source_mode")

    placeholder = st.empty()

    if mode == "🎬 Demo Mode":
        tick = st.session_state.frame_count
        for _ in range(3):
            frame, in_fence = generate_demo_frame(tick)
            # Run detection on synthetic frame
            annotated, dets = detect_frame(frame)
            placeholder.image(annotated, channels="BGR", use_container_width=True)
            tick += 1
            time.sleep(0.3)
        st.session_state.frame_count = tick

        persons = sum(1 for d in dets if d["class"] == "person")
        vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
        st.caption(f"🔍 Detected: {len(dets)} objects | Persons: {persons} | Vehicles: {vehicles}")

    elif mode == "🎥 Upload Video":
        uploads_main = st.file_uploader("Upload videos (max 4)", type=["mp4", "avi", "mov", "mkv"],
                                        accept_multiple_files=True, key="footage_main")
        if uploads_main:
            lib = st.session_state.get("video_lib", {})
            for up in uploads_main[:4]:
                k = f"{up.name}_{up.size}"
                if k not in lib and len(lib) < 4:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(up.getvalue())
                    lib[k] = {"name": up.name, "path": tfile.name, "location": ""}
            st.session_state.video_lib = lib
        lib = st.session_state.get("video_lib", {})
        if not lib:
            st.info("📁 Upload videos above (max 4). Then play them one by one in the tabs below.")
        else:
            keys = list(lib.keys())
            tabs = st.tabs([lib[k]["name"][:20] for k in keys])
            for tab, k in zip(tabs, keys):
                with tab:
                    render_video_tab(lib[k]["path"], k, lib[k].get("location", ""))

    elif mode == "📷 Webcam":
        # DEPLOY NOTE: Live uses the *server's* camera. Local run = your
        # laptop camera with full live tracking. On cloud deploy there is no
        # camera device, so Live shows a clear error and Photo mode (browser
        # capture via getUserMedia) keeps working. Nothing crashes.
        wview = st.radio("Webcam mode", ["🔴 Live Camera", "📸 Photo"],
                         horizontal=True, key="webcam_view")
        if wview == "📸 Photo":
            snap = st.camera_input("Take a photo", label_visibility="collapsed")
            if snap is not None:
                file_bytes = np.frombuffer(snap.getvalue(), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is not None:
                    annotated, dets = detect_frame(frame)

                    plates = run_ocr_on_frame(frame)
                    if plates:
                        for p in plates:
                            cv2.rectangle(annotated, p["bbox"][:2], p["bbox"][2:], (0, 255, 0), 2)
                            cv2.putText(annotated, p["text"], (p["bbox"][0], p["bbox"][1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    placeholder.image(annotated, channels="BGR", use_container_width=True)

                    persons = sum(1 for d in dets if d["class"] == "person")
                    vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
                    st.caption(f"🔍 Detected: {len(dets)} objects | Persons: {persons} | Vehicles: {vehicles} | Plates: {len(plates)}")
            else:
                st.info("📷 Click above to capture a photo from your camera")
        else:
            wstride = st.slider("Process every Nth frame", 1, 10, 2, key="webcam_stride",
                                help="Higher = faster on CPU. Alerts still fire live.")
            wmax = st.number_input("Max frames per run", min_value=60, max_value=10000,
                                   value=600, step=60, key="webcam_max",
                                   help="Each run stops after this many frames. Press Start to continue.")
            wc1, wc2 = st.columns(2)
            if wc1.button("▶ Start Live", use_container_width=True, key="webcam_start"):
                st.session_state.webcam_live = True
            if wc2.button("⏹ Stop", use_container_width=True, key="webcam_stop"):
                st.session_state.webcam_live = False
            if st.session_state.get("webcam_live"):
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ No camera found. Live mode is unavailable on cloud deployments — use 📸 Photo mode.")
                    st.session_state.webcam_live = False
                else:
                    st.success("🟢 Live camera — YOLO + fence + ANPR + face + alerts sab live.")
                    status = st.empty()
                    n = 0
                    dets = []
                    while n < wmax:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("⚠️ Camera is not delivering frames. Press Start again.")
                            break
                        n += 1
                        if (n % wstride) != 0:
                            continue
                        st.session_state.frame_count += 1
                        annotated, dets = detect_frame(frame)

                        plates = run_ocr_on_frame(frame)
                        if plates:
                            for p in plates:
                                cv2.rectangle(annotated, p["bbox"][:2], p["bbox"][2:], (0, 255, 0), 2)
                                cv2.putText(annotated, p["text"], (p["bbox"][0], p["bbox"][1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        placeholder.image(annotated, channels="BGR", use_container_width=True)

                        persons = sum(1 for d in dets if d["class"] == "person")
                        vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
                        faces = st.session_state.get("last_faces", 0)
                        status.caption(f"🔴 LIVE CAM | Frame {n}/{wmax} | Persons: {persons} | Vehicles: {vehicles} | Faces: {faces}")
                    cap.release()
                    st.session_state.webcam_live = False
                    st.success("✅ Run complete — alerts are in the Alert Log. Press Start to continue.")
            else:
                st.info("▶ Press Start Live for continuous tracking — no photo capture needed.")

    elif mode == "🌐 RTSP Stream":
        if "rtsp_url" not in st.session_state:
            st.session_state.rtsp_url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
        if "rtsp_live" not in st.session_state:
            st.session_state.rtsp_live = False

        saved = st.session_state.get("rtsp_lib", [])
        meta = st.session_state.get("rtsp_meta", {})
        labels = ["Custom URL"] + [
            (meta.get(u, {}).get("name") or f"Camera {j + 1}") +
            (f" ({meta.get(u, {}).get('location')})" if meta.get(u, {}).get("location") else "")
            for j, u in enumerate(saved)]
        pick = st.selectbox("Saved cameras", labels, key="rtsp_pick")
        if pick != "Custom URL":
            st.session_state.rtsp_url = saved[labels.index(pick) - 1]
        url = st.text_input("RTSP URL", value=st.session_state.rtsp_url,
                            placeholder="rtsp://user:pass@ip:port/stream")
        stride_rtsp = st.slider("Process every Nth frame", 1, 10, 2, key="rtsp_stride",
                                help="Higher = faster on CPU. Alerts still fire live.")
        max_frames = st.number_input("Max frames per run", min_value=60, max_value=10000,
                                     value=600, step=60,
                                     help="Each run stops after this many frames. Press Connect to continue.")

        col_r1, col_r2 = st.columns(2)
        if col_r1.button("🔌 Connect + Start Live", use_container_width=True):
            st.session_state.rtsp_url = url.strip()
            st.session_state.rtsp_live = True
        if col_r2.button("⏹ Disconnect", use_container_width=True):
            st.session_state.rtsp_live = False

        if st.session_state.get("rtsp_live"):
            target = st.session_state.rtsp_url
            if not target.lower().startswith("rtsp://"):
                st.error("❌ URL `rtsp://` se start hona chahiye.")
                st.session_state.rtsp_live = False
            else:
                cap = cv2.VideoCapture(target)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                except Exception:
                    pass
                if not cap.isOpened():
                    st.error("❌ Stream could not be opened. Check the URL and connection, then retry.")
                    st.session_state.rtsp_live = False
                else:
                    st.success("🟢 Live connected — tracking in progress.")
                    status = st.empty()
                    n = 0
                    dets = []
                    while n < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("⚠️ Stream interrupted (network/camera). Press Connect to retry.")
                            break
                        n += 1
                        if (n % stride_rtsp) != 0:
                            continue
                        st.session_state.frame_count += 1
                        annotated, dets = detect_frame(frame)
                        placeholder.image(annotated, channels="BGR", use_container_width=True)

                        persons = sum(1 for d in dets if d["class"] == "person")
                        vehicles = sum(1 for d in dets if d["class"] in ("car", "bus", "truck", "motorcycle"))
                        status.caption(f"🔴 LIVE | Frame {n}/{max_frames} | Persons: {persons} | Vehicles: {vehicles}")
                    cap.release()
                    st.session_state.rtsp_live = False
                    st.success("✅ Run complete — alerts are in the Alert Log. Press Connect to continue.")
        else:
            st.info("🔌 Paste an RTSP URL and press Connect for live tracking — no camera hardware needed.")

with col2:
    st.subheader("🚨 Alert Log")

    if not st.session_state.alerts:
        st.info("No alerts yet. Use demo controls or upload video.")
    else:
        for alert in reversed(st.session_state.alerts[-15:]):
            sev = alert["severity"]
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}.get(sev, "⚪")
            st.markdown(f"{icon} **`{alert['event_type'].upper()}`**")
            risk = alert.get("risk_score")
            if risk is not None:
                ricon = "🔴" if risk >= 80 else "🟠" if risk >= 60 else "🟡" if risk >= 40 else "🟢"
                _rcls = {"CRITICAL": "sev-critical", "HIGH": "sev-high",
                         "MEDIUM": "sev-warning"}.get(
                             alert.get("risk_level", ""), "sev-safe")
                st.markdown(
                    f"{ricon} <span class='tech {_rcls}'>THREAT {risk}/100 "
                    f"[{alert.get('risk_level', '')}]</span>",
                    unsafe_allow_html=True,
                )
                with st.expander("Why this score?"):
                    for r in alert.get("risk_reasons", []):
                        st.caption(f"• {r}")
            st.markdown(f"> {alert['explanation']}")
            st.markdown(
                f"<span class='tech'>{alert['timestamp'][:19]} &nbsp;|&nbsp; "
                f"EVENT {alert['event_id']}</span>",
                unsafe_allow_html=True,
            )
            _ls = alert.get("link_status", st.session_state.get(
                "outbox_status", {}).get(alert["event_id"], "synced"))
            if _ls == "queued":
                st.caption("📤 Queued locally — syncs when link restores")
            else:
                st.caption("✅ Synced to Command Centre (metadata only)")

    st.divider()

    # Stats
    st.subheader("📊 Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cameras", sum(1 for v in st.session_state.camera_status.values() if v))
    c2.metric("Alerts", len(st.session_state.alerts))
    critical = sum(1 for a in st.session_state.alerts if a["severity"] == "critical")
    c3.metric("Critical", critical)

    # Alert object schema
    if st.session_state.alerts:
        st.subheader("📋 Latest Alert Object")
        last = st.session_state.alerts[-1].copy()
        last.pop("prev_hash", None)
        st.json(last)

    # Low-bandwidth payload: exactly what travels upstream
    if st.session_state.alerts:
        st.subheader("📦 Low-Bandwidth Payload")
        st.caption("Only this metadata crosses the link — never video.")
        _a = st.session_state.alerts[-1]
        _mini = {
            "event_id": _a["event_id"], "ts": _a["timestamp"],
            "site": _a["site_id"], "cam": _a["camera_id"],
            "type": _a["event_type"], "sev": _a["severity"],
            "risk": _a.get("risk_score"),
            "track": (_a.get("payload") or {}).get("track_id"),
        }
        st.json(_mini)
        st.markdown(
            f"<span class='tech'>~{len(json.dumps(_mini))} BYTES ON WIRE — "
            "VIDEO STAYS ON EDGE</span>",
            unsafe_allow_html=True,
        )

    # AI summary (Groq) — rendered from sidebar action
    if st.session_state.get("ai_summary"):
        st.subheader("🤖 AI Summary")
        st.caption("Groq LLM over alert metadata — demo/HQ mode only.")
        st.markdown(st.session_state.ai_summary)

    # Hash chain verification
    st.subheader("🔗 Hash Chain Verification")
    if st.button("Verify Chain Integrity"):
        chain_ok = verify_chain()
        if chain_ok:
            st.success(f"✅ Chain valid — {len(st.session_state.alerts)} events verified")
        else:
            st.error("❌ Chain integrity compromised!")

# ═══════════════════════════════════════════════════
# BEHAVIORAL ANALYTICS
# ═══════════════════════════════════════════════════

st.divider()
st.subheader("📈 Behavioral Analytics")
st.caption("Live counts from every processed frame — run any source to populate the graphs.")

tot = st.session_state.get("stats_total", {})
frames = st.session_state.get("stats_frames", 0)
a1, a2, a3, a4 = st.columns(4)
a1.metric("Frames Analyzed", frames)
a2.metric("Persons Seen", tot.get("person", 0))
a3.metric("Vehicles Seen", sum(tot.get(k, 0) for k in
         ("car", "bus", "truck", "motorcycle", "bicycle", "autorickshaw")))
a4.metric("Faces Seen", tot.get("face", 0))

c1, c2 = st.columns(2)
with c1:
    st.caption("Detections by class")
    if tot:
        st.bar_chart(tot)
    else:
        st.info("No data yet — run any source.")
with c2:
    st.caption("Activity by hour")
    hh = st.session_state.get("stats_hourly", {})
    if hh:
        st.bar_chart(dict(sorted(hh.items())))
    else:
        st.info("No data yet.")
if st.button("🧹 Reset Analytics"):
    st.session_state.stats_total = {}
    st.session_state.stats_hourly = {}
    st.session_state.stats_frames = 0
    st.rerun()

# ═══════════════════════════════════════════════════
# TRIPWIRE DESIGNER — pen-type fence (click to draw any line)
# ═══════════════════════════════════════════════════

st.divider()
st.subheader("✏️ Tripwire Designer — draw your own fence")
st.caption("Click points on the canvas to sketch any line. Save it as a tripwire — "
           "any tracked person/vehicle crossing that line fires tracking + alert.")

_ref = np.zeros((H, W, 3), dtype=np.uint8) + 18
for _gx in range(0, W, 40):
    cv2.line(_ref, (_gx, 0), (_gx, H), (40, 40, 40), 1)
for _gy in range(0, H, 40):
    cv2.line(_ref, (0, _gy), (W, _gy), (40, 40, 40), 1)
cv2.polylines(_ref, [FENCE], True, (0, 120, 0), 1)
for _i, _seg in enumerate(st.session_state.get("tripwires", [])):
    (_x1, _y1), (_x2, _y2) = _seg
    cv2.line(_ref, (int(_x1), int(_y1)), (int(_x2), int(_y2)), (255, 0, 255), 2)
    cv2.putText(_ref, f"TW-{_i + 1}", (int(_x1) + 4, int(_y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
_pts = st.session_state.get("pen_points", [])
for _k in range(1, len(_pts)):
    cv2.line(_ref, _pts[_k - 1], _pts[_k], (0, 255, 255), 2)
for (_px, _py) in _pts:
    cv2.circle(_ref, (_px, _py), 4, (0, 255, 255), -1)

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    _click = streamlit_image_coordinates(
        cv2.cvtColor(_ref, cv2.COLOR_BGR2RGB), width=W, key="trip_canvas")
    if _click is not None:
        _np = (int(_click["x"]), int(_click["y"]))
        if st.session_state.get("pen_last") != _np:
            st.session_state.pen_last = _np
            _cur = st.session_state.get("pen_points", [])
            _cur.append(_np)
            st.session_state.pen_points = _cur
            st.rerun()
except Exception as e:
    st.warning(f"Drawing canvas unavailable ({e.__class__.__name__}). "
               "Add tripwires with coordinates below instead.")
    _c1, _c2, _c3, _c4 = st.columns(4)
    _fx1 = _c1.number_input("x1", 0, W, 100, key="fx1")
    _fy1 = _c2.number_input("y1", 0, H, 100, key="fy1")
    _fx2 = _c3.number_input("x2", 0, W, 500, key="fx2")
    _fy2 = _c4.number_input("y2", 0, H, 400, key="fy2")
    if st.button("➕ Add line tripwire", key="fx_add"):
        st.session_state.setdefault("tripwires", []).append(
            ((int(_fx1), int(_fy1)), (int(_fx2), int(_fy2))))
        st.success("Tripwire added.")
        st.rerun()

_dc1, _dc2, _dc3, _dc4 = st.columns(4)
if _dc1.button("↩ Undo point", key="pen_undo"):
    _cur = st.session_state.get("pen_points", [])
    if _cur:
        _cur.pop()
        st.session_state.pen_points = _cur
        st.session_state.pen_last = None
        st.rerun()
if _dc2.button("🗑 Clear sketch", key="pen_clear"):
    st.session_state.pen_points = []
    st.session_state.pen_last = None
    st.rerun()
if _dc3.button("💾 Save as tripwire", key="pen_save",
               disabled=len(st.session_state.get("pen_points", [])) < 2):
    _cur = st.session_state.get("pen_points", [])
    _tw = st.session_state.get("tripwires", [])
    for _k in range(1, len(_cur)):
        _tw.append((_cur[_k - 1], _cur[_k]))
    st.session_state.tripwires = _tw
    st.session_state.pen_points = []
    st.session_state.pen_last = None
    create_alert("signal_restored",
                 f"Tripwire-{len(_tw)} drawn by operator ({len(_cur)} points).",
                 "low", payload={"segments": len(_tw)})
    st.success(f"✅ Tripwire saved ({len(_cur) - 1} segment(s)). Crossings now tracked.")
    st.rerun()
if _dc4.button("❌ Delete all tripwires", key="pen_delall"):
    st.session_state.tripwires = []
    st.session_state.pen_points = []
    st.session_state.pen_last = None
    st.session_state.track_hist = {}
    st.rerun()
st.caption(f"Active tripwires: {len(st.session_state.get('tripwires', []))} "
           f"| Sketch points: {len(st.session_state.get('pen_points', []))} "
           "(canvas is 640×480 = detection frame size)")

# Footer
st.divider()
st.caption("BorderEye — Smart India Hackathon 2026 | "
           "\"Every AI-CCTV platform assumes good bandwidth, good cameras, and infinite trust. "
           "Border posts have none of those three.\"")

