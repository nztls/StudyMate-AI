from __future__ import annotations

from typing import Any, Dict, List
import io
import json
import os
import re
from pathlib import Path

from openai import AzureOpenAI
from search import search_documents

# Optional Azure Speech SDK
try:
    import azure.cognitiveservices.speech as speechsdk  # type: ignore
except Exception:
    speechsdk = None

from gtts import gTTS

DEFAULT_AZURE_OPENAI_API_VERSION = "2024-02-15-preview"


# ------------------ Azure Client ------------------ #

def _deployment_name() -> str:
    dep = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if not dep:
        raise ValueError("Eksik ortam değişkeni: AZURE_OPENAI_DEPLOYMENT")
    return dep


def _get_api_version() -> str:
    return (
        os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
        or os.getenv("OPENAI_API_VERSION", "").strip()
        or DEFAULT_AZURE_OPENAI_API_VERSION
    )


def _require_azure_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

    missing = [k for k, v in [
        ("AZURE_OPENAI_ENDPOINT", endpoint),
        ("AZURE_OPENAI_API_KEY", api_key),
        ("AZURE_OPENAI_DEPLOYMENT", deployment),
    ] if not v]

    if missing:
        raise ValueError(f"Eksik ortam değişkenleri: {', '.join(missing)}")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=_get_api_version(),
    )


# ------------------ Parsing Helpers ------------------ #

def _json_cleanup(text: str) -> str:
    """
    Common cleanup for model outputs that try to be JSON but aren't.
    Used only as fallback when JSON mode isn't available.
    """
    s = (text or "").strip()

    # smart quotes -> normal quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    return s


def _safe_json_loads(s: str) -> Any:
    s = _json_cleanup(s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # try extract json object/array from within the output
        a0, a1 = s.find("["), s.rfind("]")
        o0, o1 = s.find("{"), s.rfind("}")
        if a0 != -1 and a1 != -1 and a1 > a0:
            return json.loads(_json_cleanup(s[a0:a1 + 1]))
        if o0 != -1 and o1 != -1 and o1 > o0:
            return json.loads(_json_cleanup(s[o0:o1 + 1]))
        raise


def _looks_turkish(text: str) -> bool:
    return bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", text or ""))


def _pick_lang_mode(language: str, text_hint: str) -> str:
    language = (language or "auto").lower().strip()
    if language in ("tr", "en"):
        return language
    return "tr" if _looks_turkish(text_hint) else "en"


def _chat_complete_text(messages: List[Dict], temperature: float = 0.2, max_tokens: int = 700) -> str:
    client = _require_azure_client()
    resp = client.chat.completions.create(
        model=_deployment_name(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _chat_complete_json(messages: List[Dict], temperature: float = 0.2, max_tokens: int = 700) -> Any:
    """
    Best effort JSON:
    1) Try JSON mode (response_format=json_object) -> should be valid JSON object
    2) If Azure/model/api_version doesn't support it, fall back to text+robust parsing
    """
    client = _require_azure_client()

    # 1) JSON mode attempt
    try:
        resp = client.chat.completions.create(
            model=_deployment_name(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        return json.loads(content)  # should be valid
    except Exception:
        # 2) fallback
        raw = _chat_complete_text(messages, temperature=temperature, max_tokens=max_tokens)
        return _safe_json_loads(raw)


def _coerce_slide_obj(obj: Any, fallback_title: str) -> Dict[str, Any]:
    """
    Normalize slide output to:
      {"title": str, "bullets": List[str]}
    Handles dict, [dict], [str], weird types.
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return {"title": fallback_title, "bullets": []}
        if isinstance(obj[0], dict):
            obj = obj[0]
        elif all(isinstance(x, str) for x in obj):
            return {"title": fallback_title, "bullets": [x.strip() for x in obj if x.strip()]}
        else:
            return {"title": fallback_title, "bullets": [str(x).strip() for x in obj if str(x).strip()]}

    if isinstance(obj, dict):
        title = (obj.get("title") or obj.get("başlık") or obj.get("baslik") or fallback_title)
        title = str(title).strip() if title is not None else fallback_title

        bullets = obj.get("bullets") or obj.get("maddeler") or obj.get("points") or obj.get("bullet_points") or []

        if isinstance(bullets, str):
            lines = [ln.strip("-• \t").strip() for ln in bullets.splitlines()]
            bullets = [ln for ln in lines if ln]

        if not isinstance(bullets, list):
            bullets = [str(bullets).strip()] if str(bullets).strip() else []

        cleaned: List[str] = []
        for b in bullets:
            if b is None:
                continue
            s = str(b).strip()
            if s:
                cleaned.append(s)

        return {"title": title, "bullets": cleaned}

    s = str(obj).strip()
    return {"title": fallback_title, "bullets": [s]} if s else {"title": fallback_title, "bullets": []}


# ------------------ RAG Q&A ------------------ #

def answer_with_rag(user_message: str, language: str = "auto") -> str:
    user_message = (user_message or "").strip()
    if not user_message:
        return ""

    hits = search_documents(user_message, top=7)
    if not hits:
        lang = _pick_lang_mode(language, user_message)
        return (
            "Henüz bir ders dokümanı yüklenmedi veya indekslenmedi. Lütfen önce dosya yükle."
            if lang == "tr"
            else "No course document is indexed yet. Please upload a document first."
        )

    strong = [(c, s) for (c, s) in hits if s >= 0.22]
    if strong:
        hits = strong

    context = "\n\n".join([c for c, _ in hits])[:7000]
    lang = _pick_lang_mode(language, user_message)

    if lang == "tr":
        sys = (
            "Sen bir üniversite ders asistanısın. "
            "SADECE verilen ders notu parçalarına (CONTEXT) dayan. "
            "CONTEXT'te yoksa uydurma; 'Dokümanda geçmiyor' de. "
            "Cevabı öğretici yaz; kritik noktaları vurgula."
        )
        user = f"""
Aşağıdaki CONTEXT, öğrencinin yüklediği ders dokümanından alınmıştır.
Sadece buna dayanarak soruyu yanıtla.

Çıktı formatı (başlıklar aynen):
Tanım:
Kritik Noktalar:
- ...
- ...
- ...
Örnek / Sezgi (varsa):
Yaygın Hata:
Kısa Özet (1 cümle):

CONTEXT:
{context}

SORU:
{user_message}
"""
    else:
        sys = (
            "You are a university teaching assistant. "
            "Use ONLY the given course excerpts (CONTEXT). "
            "If it isn't in CONTEXT, say 'Not covered in the document' and do not invent. "
            "Be teaching-focused and highlight key points."
        )
        user = f"""
CONTEXT is extracted from the student's uploaded course document.
Answer using ONLY CONTEXT.

Exact format:
Definition:
Key Points:
- ...
- ...
- ...
Example / Intuition (if any):
Common Mistake:
One-sentence Summary:

CONTEXT:
{context}

QUESTION:
{user_message}
"""

    return _chat_complete_text(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=750,
    )


# ------------------ Summary (JSON Mode) ------------------ #

def generate_summary(raw_text: str, language: str = "auto") -> Dict:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {}

    lang = _pick_lang_mode(language, raw_text[:1200])
    snippet = raw_text[:14000]

    if lang == "tr":
        sys = "Sen bir üniversite ders asistanısın. Sadece verilen ders metnine dayan. Uydurma yapma."
        user = f"""
Aşağıdaki ders metninden, öğrencinin hızlı çalışmasına uygun bir özet çıkar.

SADECE geçerli JSON döndür (başka hiçbir şey yazma).
Format:
{{
  "genel_ozet": "8-12 cümle öğretici anlatım",
  "kilit_kavramlar": ["..."],
  "kilit_noktalar": ["..."],
  "yanlis_anlasilmalar": ["..."],
  "calisma_plani": ["...","...","..."]
}}

DERS METNİ:
{snippet}
"""
    else:
        sys = "You are a university teaching assistant. Use only the given text. Do not invent."
        user = f"""
Create a study-friendly summary from the course text below.

Return ONLY valid JSON (no extra text).
Format:
{{
  "overview": "8-12 sentence teaching-style explanation",
  "key_concepts": ["..."],
  "key_points": ["..."],
  "misconceptions": ["..."],
  "study_plan": ["...","...","..."]
}}

COURSE TEXT:
{snippet}
"""

    obj = _chat_complete_json(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=900,
    )

    return obj if isinstance(obj, dict) else {}


# ------------------ Slides (JSON Mode) ------------------ #

def _generate_outline_titles(raw_text: str, max_slides: int, language: str) -> List[str]:
    snippet = (raw_text or "").strip()[:9000]
    if not snippet:
        return ["Genel Özet"] if language == "tr" else ["Overview"]

    if language == "tr":
        sys = "Sen ders içeriğinden sunum başlığı çıkaran bir asistansın."
        user = f"""
Aşağıdaki ders metninden {max_slides} adet sunum slayt başlığı üret.
Kurallar:
- Başlıklar konu bazlı, net ve özgül olsun.
- Her satıra 1 başlık yaz.
- Gereksiz numaralandırma yapma.

METİN:
{snippet}
"""
    else:
        sys = "You generate presentation slide titles from course text."
        user = f"""
Create {max_slides} presentation slide titles from the text.
Rules:
- Titles must be topic-based and specific.
- One title per line.
- No unnecessary numbering.

TEXT:
{snippet}
"""

    raw = _chat_complete_text(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=220,
    )

    titles: List[str] = []
    for line in raw.splitlines():
        t = re.sub(r"^[-•\s]*", "", line.strip())
        t = re.sub(r"^\d+[\).\s]+", "", t).strip()
        if len(t) >= 4:
            titles.append(t)

    return titles[:max_slides] if titles else (["Genel Özet"] if language == "tr" else ["Overview"])


def generate_slides_from_text(raw_text: str, max_slides: int = 8, language: str = "auto") -> List[Dict]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return []

    lang = _pick_lang_mode(language, raw_text[:1200])
    titles = _generate_outline_titles(raw_text, max_slides=max_slides, language=lang)

    slides: List[Dict] = []

    for title in titles:
        hits = search_documents(title, top=8)
        strong = [(c, s) for (c, s) in hits if s >= 0.20]
        if strong:
            hits = strong
        context = "\n\n".join([c for c, _ in hits])[:7000] if hits else raw_text[:7000]

        if lang == "tr":
            sys = "Sen ders anlatımı için slayt hazırlayan bir asistansın. Sadece verilen NOTLAR'a dayan. Uydurma yapma."
            user = f"""
Aşağıdaki NOTLAR'a dayanarak 1 adet slayt üret.

Kurallar:
- Başlık kısa ve konu-odaklı olsun.
- 4–6 bullet yaz (her bullet 6–14 kelime arası).
- Bulletlar tanım tekrar etmesin; kritik öğrenme noktası olsun.
- En sona 1 satır "Sınav İpucu:" ekle.
- SADECE geçerli JSON döndür.

JSON formatı:
{{"title":"...","bullets":["...","...","..."]}}

İstenen konu: {title}

NOTLAR:
{context}
"""
        else:
            sys = "You create presentation slides from course notes. Use only the given NOTES. Do not invent."
            user = f"""
Using ONLY the NOTES below, create exactly ONE slide.

Rules:
- Title short and topic-focused.
- 4–6 bullets, each 6–14 words.
- Bullets must be key learning points, not repetitive definitions.
- End with one line starting with 'Exam Tip:'.
- Return ONLY valid JSON.

JSON format:
{{"title":"...","bullets":["...","...","..."]}}

Topic: {title}

NOTES:
{context}
"""

        obj = _chat_complete_json(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=450,
        )

        slide = _coerce_slide_obj(obj, fallback_title=title)

        bullets = slide.get("bullets", [])
        if isinstance(bullets, list):
            slide["bullets"] = bullets[:10]

        slides.append(slide)

    return slides[:max_slides]


# ------------------ TTS ------------------ #

def tts_bytes(text: str, language: str = "auto") -> bytes:
    text = (text or "").strip()
    if not text:
        return b""

    lang = _pick_lang_mode(language, text[:500])
    speech_key = os.getenv("AZURE_SPEECH_KEY", "").strip()
    speech_region = os.getenv("AZURE_SPEECH_REGION", "").strip()

    if speechsdk is not None and speech_key and speech_region and speech_key != ".":
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        speech_config.speech_synthesis_language = "tr-TR" if lang == "tr" else "en-US"

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return bytes(result.audio_data)

    tts_lang = "tr" if lang == "tr" else "en"
    fp = io.BytesIO()
    gTTS(text=text, lang=tts_lang).write_to_fp(fp)
    fp.seek(0)
    return fp.read()


# ------------------ PDF Export (Unicode Font Fix) ------------------ #

def _find_unicode_ttf() -> Path | None:
    candidates = [
        Path("fonts/DejaVuSans.ttf"),
        Path("fonts/NotoSans-Regular.ttf"),
        Path("fonts/SegoeUI.ttf"),
        Path("fonts/ArialUnicodeMS.ttf"),
    ]

    win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates += [
        win_fonts / "DejaVuSans.ttf",
        win_fonts / "dejavusans.ttf",
        win_fonts / "NotoSans-Regular.ttf",
        win_fonts / "NotoSans.ttf",
        win_fonts / "segoeui.ttf",
        win_fonts / "arial.ttf",
        win_fonts / "Arial.ttf",
        win_fonts / "ARIALUNI.TTF",
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def build_summary_pdf_bytes(summary: Dict, slides: List[Dict], language: str = "auto", title: str = "Course Summary") -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, PageBreak
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    ttf_path = _find_unicode_ttf()
    font_name = "Helvetica"
    if ttf_path is not None:
        font_name = "UnicodeFont"
        try:
            pdfmetrics.registerFont(TTFont(font_name, str(ttf_path)))
        except Exception:
            font_name = "Helvetica"

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    base_styles = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("TitleUnicode", parent=base_styles["Title"], fontName=font_name),
        "Heading2": ParagraphStyle("Heading2Unicode", parent=base_styles["Heading2"], fontName=font_name),
        "Heading3": ParagraphStyle("Heading3Unicode", parent=base_styles["Heading3"], fontName=font_name),
        "BodyText": ParagraphStyle("BodyUnicode", parent=base_styles["BodyText"], fontName=font_name, leading=14),
    }

    story = []
    hint = (summary.get("genel_ozet") or summary.get("overview") or "")[:200]
    lang = _pick_lang_mode(language, hint)

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    def _add_list(h, items):
        items = items or []
        if not items:
            return
        story.append(Paragraph(f"<b>{h}</b>", styles["Heading2"]))
        lf = ListFlowable([ListItem(Paragraph(str(x), styles["BodyText"])) for x in items], bulletType="bullet")
        story.append(lf)
        story.append(Spacer(1, 10))

    if lang == "tr":
        genel = summary.get("genel_ozet", "")
        if genel:
            story.append(Paragraph("<b>Genel Özet</b>", styles["Heading2"]))
            story.append(Paragraph(genel, styles["BodyText"]))
            story.append(Spacer(1, 10))

        _add_list("Kilit Kavramlar", summary.get("kilit_kavramlar"))
        _add_list("Kilit Noktalar", summary.get("kilit_noktalar"))
        _add_list("Yanlış Anlaşılmalar", summary.get("yanlis_anlasilmalar"))
        _add_list("Çalışma Planı", summary.get("calisma_plani"))

        story.append(PageBreak())
        story.append(Paragraph("<b>Slayt Özeti</b>", styles["Heading2"]))
        story.append(Spacer(1, 10))
    else:
        overview = summary.get("overview", "")
        if overview:
            story.append(Paragraph("<b>Overview</b>", styles["Heading2"]))
            story.append(Paragraph(overview, styles["BodyText"]))
            story.append(Spacer(1, 10))

        _add_list("Key Concepts", summary.get("key_concepts"))
        _add_list("Key Points", summary.get("key_points"))
        _add_list("Misconceptions", summary.get("misconceptions"))
        _add_list("Study Plan", summary.get("study_plan"))

        story.append(PageBreak())
        story.append(Paragraph("<b>Slide Summary</b>", styles["Heading2"]))
        story.append(Spacer(1, 10))

    for i, s in enumerate(slides or [], start=1):
        if not isinstance(s, dict):
            s = {"title": f"Slide {i}", "bullets": [str(s)]}

        stitle = (s.get("title") or f"Slide {i}").strip()
        bullets = s.get("bullets") or []
        if not isinstance(bullets, list):
            bullets = [str(bullets)]

        story.append(Paragraph(f"{i}. {stitle}", styles["Heading3"]))
        if bullets:
            lf = ListFlowable([ListItem(Paragraph(str(b), styles["BodyText"])) for b in bullets], bulletType="bullet")
            story.append(lf)
        story.append(Spacer(1, 10))

    doc.build(story)
    buf.seek(0)
    return buf.read()
