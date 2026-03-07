import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from search import build_index_from_text, has_index
from agent import (
    answer_with_rag,
    generate_summary,
    generate_slides_from_text,
    tts_bytes,
    build_summary_pdf_bytes,
)

load_dotenv()

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="wide")
st.title("🎓 AI Tutor")

st.write(
    "1) Soldan ders dokümanını yükle (PDF/TXT)\n"
    "2) **Chat** sekmesinde dokümana dayalı soru sor\n"
    "3) **Özet** sekmesinde slayt gibi özet al, sesli dinle ve PDF indir"
)

if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""
if "chat" not in st.session_state:
    st.session_state.chat = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "slides" not in st.session_state:
    st.session_state.slides = None
if "audio" not in st.session_state:
    st.session_state.audio = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

st.sidebar.header("📚 Ders Dokümanı Yükle")

lang_mode = st.sidebar.radio("Dil modu", ["Auto", "Türkçe", "English"], index=0)
LANG = "auto"
if lang_mode == "Türkçe":
    LANG = "tr"
elif lang_mode == "English":
    LANG = "en"

uploaded_file = st.sidebar.file_uploader(
    "PDF veya TXT yükle",
    type=["pdf", "txt"],
    accept_multiple_files=False,
)

def _extract_text(uploaded) -> str:
    name = uploaded.name.lower()

    if name.endswith(".txt") or (uploaded.type or "").startswith("text/"):
        data = uploaded.read()
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")

    reader = PdfReader(uploaded)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages)

if uploaded_file is not None:
    st.session_state.doc_name = uploaded_file.name
    with st.spinner("Dosya okunuyor..."):
        text = _extract_text(uploaded_file)

    if not text.strip():
        st.sidebar.error(
            "Bu PDF tarama (scan) olabilir veya metin çıkarılamadı. "
            "Metin tabanlı PDF/TXT deneyebilirsin."
        )
    else:
        st.session_state.doc_text = text
        build_index_from_text(text)
        st.sidebar.success("Doküman yüklendi ve indekslendi ✅")
else:
    if not has_index():
        st.sidebar.info("Devam etmek için bir doküman yükle.")

tab_chat, tab_summary = st.tabs(["💬 Chat (Dokümana Dayalı)", "📄 Özet + Slayt + Ses + PDF"])

with tab_chat:
    st.subheader("Dokümana Dayalı Chat")

    if not has_index():
        st.warning("Önce soldan bir PDF/TXT yüklemen gerekiyor.")
    else:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        user_input = st.chat_input("Sorunu yaz sor 😄")
        if user_input:
            st.session_state.chat.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Cevap hazırlanıyor..."):
                    try:
                        ans = answer_with_rag(user_input, language=LANG)
                    except Exception as e:
                        ans = f"Hata: {e}"
                    st.write(ans)

            st.session_state.chat.append({"role": "assistant", "content": ans})

with tab_summary:
    st.subheader("Dokümanı Özetle → Slayt gibi çıkar → Seslendir → PDF indir")

    if not has_index() or not st.session_state.doc_text.strip():
        st.warning("Önce soldan bir doküman yüklemen gerekiyor.")
    else:
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            max_slides = st.slider("Doküman kaç başlıkta özetlenecek?", min_value=4, max_value=15, value=8)
        with colB:
            do_audio = st.checkbox("Ses de hazırla", value=True)
        with colC:
            pdf_title = st.text_input(
                "PDF başlığı",
                value=f"Course Summary - {st.session_state.doc_name}" if st.session_state.doc_name else "Course Summary",
            )

        if st.button("🚀 Özet + Slayt Üret", type="primary"):
            st.session_state.audio = None
            st.session_state.pdf_bytes = None

            with st.spinner("Özet çıkarılıyor (Azure OpenAI)..."):
                try:
                    summ = generate_summary(st.session_state.doc_text, language=LANG)
                except Exception as e:
                    st.error(f"Özet hatası: {e}")
                    summ = None

            with st.spinner("Slayt özeti hazırlanıyor (Azure OpenAI)..."):
                try:
                    slides = generate_slides_from_text(
                        st.session_state.doc_text,
                        max_slides=max_slides,
                        language=LANG,
                    )
                except Exception as e:
                    st.error(f"Slayt hatası: {e}")
                    slides = None

            st.session_state.summary = summ
            st.session_state.slides = slides

            if summ and slides:
                with st.spinner("PDF hazırlanıyor..."):
                    try:
                        st.session_state.pdf_bytes = build_summary_pdf_bytes(
                            summary=summ,
                            slides=slides,
                            language=LANG,
                            title=pdf_title.strip() or "Course Summary",
                        )
                    except Exception as e:
                        st.error(f"PDF hatası: {e}")

            if do_audio and summ:
                with st.spinner("Ses hazırlanıyor..."):
                    try:
                        if (LANG == "tr") or (LANG == "auto" and isinstance(summ, dict) and "genel_ozet" in summ):
                            base = summ.get("genel_ozet", "")
                            kp = (summ.get("kilit_noktalar") or [])[:6]
                            audio_text = base + "\n\nKilit Noktalar:\n" + "\n".join(f"- {x}" for x in kp)
                        else:
                            base = summ.get("overview", "")
                            kp = (summ.get("key_points") or [])[:6]
                            audio_text = base + "\n\nKey Points:\n" + "\n".join(f"- {x}" for x in kp)

                        st.session_state.audio = tts_bytes(audio_text, language=LANG)
                    except Exception as e:
                        st.error(f"Ses hatası: {e}")

            st.success("Hazır ✅")

        summ = st.session_state.summary
        slides = st.session_state.slides

        if summ:
            st.markdown("### 🧠 Özet (Öğretici Anlatım)")

            def _show_list(title: str, items, limit: int = 10):
                items = items or []
                if not items:
                    return
                st.markdown(f"**{title}**")
                for x in items[:limit]:
                    st.markdown(f"- {x}")

            if isinstance(summ, dict) and "genel_ozet" in summ:
                st.write(summ.get("genel_ozet", ""))
                _show_list("Kilit Kavramlar", summ.get("kilit_kavramlar"))
                _show_list("Kilit Noktalar", summ.get("kilit_noktalar"))
                _show_list("Yanlış Anlaşılmalar", summ.get("yanlis_anlasilmalar"))
                _show_list("Mini Çalışma Planı", summ.get("calisma_plani"), limit=5)
            elif isinstance(summ, dict):
                st.write(summ.get("overview", ""))
                _show_list("Key Concepts", summ.get("key_concepts"))
                _show_list("Key Points", summ.get("key_points"))
                _show_list("Misconceptions", summ.get("misconceptions"))
                _show_list("Study Plan", summ.get("study_plan"), limit=5)

        if slides:
            st.markdown("### 🗂️ Doküman Özeti")
            for i, s in enumerate(slides, start=1):
                if not isinstance(s, dict):
                    s = {"title": f"Slide {i}", "bullets": [str(s)]}
                st.markdown(f"#### {i}. {s.get('title','')}")
                for b in (s.get("bullets") or []):
                    st.markdown(f"- {b}")

        if st.session_state.audio:
            st.markdown("### 🔊 Sesli Okuma")
            st.audio(st.session_state.audio, format="audio/mp3")

        if st.session_state.pdf_bytes:
            st.markdown("### 📥 PDF İndir")
            st.download_button(
                label="PDF'i indir",
                data=st.session_state.pdf_bytes,
                file_name=(pdf_title.strip() or "course_summary") + ".pdf",
                mime="application/pdf",
            )
