import os
import re
from io import BytesIO
from datetime import datetime

import streamlit as st
from openai import OpenAI, OpenAIError   # noqa: F401  (kept for completeness)

# ─────────────────────────────────────────  CONFIG  ──────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Set it in your environment or Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(
    page_title="Text Differentiator Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────  CSS  ────────────────────────────────────────────
st.markdown(
    """
<style>
/* ——— Fonts & Base ——— */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*{font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','SF Pro Text','Helvetica Neue',Helvetica,Arial,sans-serif;
  box-sizing:border-box;}

/* Light / dark automatic */
.stApp{color-scheme:light dark;}

/* Hide Streamlit default chrome */
#MainMenu,footer,.viewerBadge_link__1S137{display:none;}

/* ——— Tabs ——— */
.stTabs [data-baseweb="tab-list"]{gap:24px;border-bottom:none!important;}
.stTabs [data-baseweb="tab"]{height:44px;background:transparent;border:none;font-size:17px;font-weight:500;padding-bottom:12px;}
.stTabs [aria-selected="true"]{border-bottom:none!important;}

/* ——— Inputs & Textarea ——— */
.stTextArea textarea,
.stSelectbox > div > div{
    background:var(--secondary-background-color);
    border:1px solid #d2d2d7;
    border-radius:8px;
    font-size:15px;
    color:var(--text-color)!important;   /* readable text */
}
.stTextArea textarea:focus,
input:focus,
select:focus,
textarea:focus{
    border-color:#8e8e93!important;     /* neutral grey */
    box-shadow:none!important;          /* remove red glow */
    outline:none!important;
}

/* ——— Metric cards ——— */
.metric-card{background:var(--background-color);border:1px solid #d2d2d7;border-radius:12px;padding:20px;text-align:center;}
.metric-value{font-size:32px;font-weight:700;margin-bottom:4px;}
.metric-label{font-size:13px;color:#86868b;font-weight:500;}

/* ——— Buttons ——— */
.stButton button,.stDownloadButton button{
    background:#1d1d1f;color:#fff;border:none;border-radius:8px;padding:8px 20px;font-size:15px;font-weight:500;min-height:40px;
}
.stButton button:hover,.stDownloadButton button:hover{opacity:.85;}

/* ——— Responsive tweaks ——— */
@media(max-width:768px){
  .hero-title{font-size:40px;}
  .hero-subtitle{font-size:18px;}
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────  HELPERS  ────────────────────────────────────────────
def count_syllables(word: str) -> int:
    vowels = "aeiouy"
    word = word.lower()
    count = 0
    if word and word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    return max(count, 1)


def readability(text: str):
    sentences = re.split(r"[.!?]+", text)
    words = text.split()
    if not sentences or not words:
        return None
    syllables = sum(count_syllables(w) for w in words)
    asl = len(words) / len(sentences)
    asw = syllables / len(words)
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return {
        "word_count": len(words),
        "avg_sentence_length": round(asl, 1),
        "reading_ease": round(score, 1),
    }


_GUIDES = {
    "Kindergarten": ("3–5 words", "Basic sight words", "Simple S‑V", "Concrete objects"),
    "1st Grade": ("5–8 words", "Sight + simple descript.", "Basic conj.", "Familiar experiences"),
    "2nd Grade": ("8–12 words", "Growing sight list", "and/but compounds", "Comparisons, sequence"),
    "3rd Grade": ("10–15 words", "Academic vocab", "Dep. clauses", "Abstract ideas + examples"),
    "4th Grade": ("12–18 words", "Subject terms", "Varied structs", "Cause–effect, inference"),
    "5th Grade": ("15–20 words", "Figurative language", "Sophisticated variety", "Abstract, critical"),
}
def guide(grade):  # default for 6‑12
    return _GUIDES.get(
        grade,
        ("Varies", "Grade academic vocab", "Full range", "Abstract / complex"),
    )


def history_pdf(records):
    """Return PDF bytes or None if ReportLab unavailable."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return None
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Text Differentiator Pro — History")
    y -= 30
    c.setFont("Helvetica", 11)
    for r in records:
        for line in (
            f"{r['timestamp']}  |  {r['grade']}",
            f"Original: {r['original']}",
            f"Adapted:  {r['adapted']}",
            "-" * 94,
        ):
            if y < 80:
                c.showPage()
                y = h - 50
                c.setFont("Helvetica", 11)
            c.drawString(50, y, line)
            y -= 14
    c.save()
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────── UI ──────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center;padding:3rem 0 2rem 0;">
  <h1 class="hero-title" style="margin:0;font-weight:700;">Text Differentiator Pro</h1>
  <p class="hero-subtitle" style="margin:0;">Transform instructional materials into accessible texts for every learner.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---- Session state defaults ----
st.session_state.setdefault("adapted", "")
st.session_state.setdefault("questions", "")
st.session_state.setdefault("history", [])

student_profiles = {
    "Jordan – ELL 6th Grade": {
        "grade_level": "6th Grade",
        "simplify_vocab": True,
        "short_paragraphs": True,
        "in_text_definitions": True,
        "breaks": True,
        "make_qs": True,
    },
    "Ava – IEP 4th Grade": {
        "grade_level": "4th Grade",
        "simplify_vocab": True,
        "short_paragraphs": True,
        "in_text_definitions": False,
        "breaks": False,
        "make_qs": True,
    },
    "Liam – Advanced 8th Grade": {
        "grade_level": "8th Grade",
        "simplify_vocab": False,
        "short_paragraphs": False,
        "in_text_definitions": False,
        "breaks": False,
        "make_qs": True,
    }
}

# ---- Sidebar ----
with st.sidebar:
    st.header("Configuration")
    GRADES = [
        "Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade",
        "5th Grade", "6th Grade", "7th Grade", "8th Grade",
        "9th Grade", "10th Grade", "11th Grade", "12th Grade",
    ]
    tgt_grade = st.selectbox("Target grade level", GRADES, index=2)

    st.header("AI Settings")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)

    st.header("Accessibility Options")
    simplify = st.checkbox("Simplify vocabulary", True)
    define   = st.checkbox("Add in‑text definitions", True)
    short_p  = st.checkbox("Short paragraphs", True)
    breaks   = st.checkbox("Add visual breaks", False)

    st.header("Output Options")
    make_qs  = st.checkbox("Generate comprehension questions", True)

    st.header("Student Profile")
    selected_profile = st.selectbox("Choose student profile", ["None"] + list(student_profiles.keys()))

    # Override settings if profile is selected
    if selected_profile != "None":
        profile = student_profiles[selected_profile]
        tgt_grade = profile["grade_level"]
        simplify = profile["simplify_vocab"]
        short_p = profile["short_paragraphs"]
        define = profile["in_text_definitions"]
        breaks = profile["breaks"]
        make_qs = profile["make_qs"]

        st.markdown("✅ **Profile applied:**")
        st.markdown(f"• **Grade Level:** {tgt_grade}")
        st.markdown(f"• Simplify Vocab: {simplify}")
        st.markdown(f"• Short Paragraphs: {short_p}")
        st.markdown(f"• In-text Definitions: {define}")
        st.markdown(f"• Visual Breaks: {breaks}")
        st.markdown(f"• Generate Questions: {make_qs}")

# ---- Tabs ----
tab_adapt, tab_metrics, tab_hist, tab_profiles = st.tabs(["Adapt Text", "Analytics", "History", "Profiles"])

# ===========  ADAPT TAB  ===========
with tab_adapt:
    st.subheader("Input text")
    text_in = st.text_area(
        "Paste or type the text you want to adapt",
        height=220,
        placeholder="Enter your text here…",
    )
    left, right = st.columns(2)
    adapt_btn = left.button("Adapt text", use_container_width=True)
    if right.button("Clear", use_container_width=True):
        for k in ("adapted", "questions"):
            st.session_state[k] = ""
        st.experimental_rerun()

    if adapt_btn and text_in.strip():
        with st.spinner(f"Adapting text for {tgt_grade} …"):
            rules = guide(tgt_grade)
            sys_prompt = f"""
You are an expert special‑education content specialist.

TARGET: {tgt_grade} students.

GUIDELINES
 • Sentence length: {rules[0]}
 • Vocabulary: {rules[1]}
 • Complexity: {rules[2]}
 • Concepts: {rules[3]}

ACCOMMODATIONS
 {'• Simplify vocabulary'            if simplify else ''}
 {'• Add definitions in parentheses' if define   else ''}
 {'• Short paragraphs (2‑3 sent.)'   if short_p  else ''}
 {'• Visual breaks between ideas'     if breaks   else ''}
 • Clear topic sentences, transitions
 • Active voice; literal language

PRESERVE
 • All key ideas, meaning, purpose

OUTPUT
 • Markdown only (no commentary)
 • **Bold** key terms
 • Bullet lists where useful
 • Short, focused paragraphs
"""
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Adapt this text for {tgt_grade}:\n\n{text_in}"},
            ]
            try:
                res = client.chat.completions.create(
                    model=model,
                    temperature=0.3,
                    messages=msgs,
                    max_tokens=2000,
                )
            except Exception as err:
                st.error(f"OpenAI error: {err}")
                raise err
            st.session_state.adapted = res.choices[0].message.content.strip()

            if make_qs:
                q_msgs = [
                    {"role": "system", "content": "Write clear comprehension questions for the given grade."},
                    {"role": "user", "content": f"Create 6 questions for {tgt_grade} students based on this text:\n\n{st.session_state.adapted}"},
                ]
                q_res = client.chat.completions.create(model=model, temperature=0.3, messages=q_msgs)
                st.session_state.questions = q_res.choices[0].message.content.strip()

            # history preview
            hist = st.session_state.history
            hist.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "grade": tgt_grade,
                    "original": (text_in[:100] + "...") if len(text_in) > 100 else text_in,
                    "adapted": (st.session_state.adapted[:100] + "...") if len(st.session_state.adapted) > 100 else st.session_state.adapted,
                }
            )
            st.success("Adaptation complete.")

    # ---- Show results ----
    if st.session_state.adapted:
        st.markdown("#### Comparison")
        st.markdown(
            f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
  <div style="border:1px solid #d2d2d7;border-radius:12px;padding:20px;height:450px;overflow:auto;">
      <h5>Original</h5>
      <div style="white-space:pre-wrap;font-size:15px;">{text_in}</div>
  </div>
  <div style="border:1px solid #d2d2d7;border-radius:12px;padding:20px;height:450px;overflow:auto;">
      <h5>Adapted for {tgt_grade}</h5>
      <div style="white-space:pre-wrap;font-size:15px;">{st.session_state.adapted}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if make_qs and st.session_state.questions:
            st.markdown("#### Comprehension questions")
            st.markdown(st.session_state.questions)

        # downloads
        st.markdown("---")
        pack = f"""GENERATED {datetime.now():%Y-%m-%d %H:%M}
GRADE {tgt_grade}
MODEL {model}

ORIGINAL
--------
{text_in}

ADAPTED
-------
{st.session_state.adapted}

{"QUESTIONS\n---------\n" + st.session_state.questions if st.session_state.questions else ""}
"""
        col1, col2, col3 = st.columns(3)
        col1.download_button(
            "Download adapted text",
            st.session_state.adapted,
            f"adapted_{tgt_grade}_{datetime.now():%Y%m%d_%H%M}.txt",
            "text/plain",
        )
        if st.session_state.questions:
            col2.download_button(
                "Download questions",
                st.session_state.questions,
                f"questions_{tgt_grade}_{datetime.now():%Y%m%d_%H%M}.txt",
                "text/plain",
            )
        col3.download_button(
            "Download complete package",
            pack,
            f"package_{tgt_grade}_{datetime.now():%Y%m%d_%H%M}.txt",
            "text/plain",
        )

# ===========  METRICS TAB  ===========
with tab_metrics:
    st.subheader("Text analytics")
    if st.session_state.adapted and text_in.strip():
        o = readability(text_in)
        a = readability(st.session_state.adapted)
        if o and a:
            cols = st.columns(4)
            cols[0].markdown(f"""<div class="metric-card"><div class="metric-value">{o['word_count']}→{a['word_count']}</div><div class="metric-label">Words</div></div>""", unsafe_allow_html=True)
            cols[1].markdown(f"""<div class="metric-card"><div class="metric-value">{o['avg_sentence_length']}→{a['avg_sentence_length']}</div><div class="metric-label">Avg sent length</div></div>""", unsafe_allow_html=True)
            cols[2].markdown(f"""<div class="metric-card"><div class="metric-value">{o['reading_ease']}→{a['reading_ease']}</div><div class="metric-label">Flesch ease</div></div>""", unsafe_allow_html=True)
            drop = round((1 - a['word_count'] / o['word_count']) * 100)
            cols[3].markdown(f"""<div class="metric-card"><div class="metric-value">{drop}%</div><div class="metric-label">Complexity ↓</div></div>""", unsafe_allow_html=True)
    else:
        st.write("Adapt a text first to view analytics.")

# ===========  HISTORY TAB  ===========
with tab_hist:
    st.subheader("Adaptation history")
    hist = st.session_state.history
    if hist:
        pdf = history_pdf(hist)
        if pdf:
            st.download_button(
                "Download history (PDF)",
                pdf,
                f"history_{datetime.now():%Y%m%d_%H%M}.pdf",
                "application/pdf",
            )
        else:
            st.caption("Install reportlab to enable PDF export.")
        for rec in reversed(hist[-10:]):
            with st.expander(f"{rec['timestamp']} — {rec['grade']}"):
                st.write("**Original preview:**", rec["original"])
                st.write("**Adapted preview:**", rec["adapted"])
    else:
        st.write("No history yet – run an adaptation first.")
# ===========  PROFILES TAB  ===========
with tab_profiles:
    st.subheader("Select and Apply Student Profile")

    selected_profile = st.selectbox("Choose student profile", ["None"] + list(student_profiles.keys()))

    if selected_profile != "None":
        profile = student_profiles[selected_profile]

        st.markdown(f"### 👤 Profile: {selected_profile}")
        st.markdown(f"**Grade Level:** {profile['grade_level']}")
        st.markdown("**Adaptation Settings:**")
        for k, v in profile.items():
            if k != "grade_level":
                label = k.replace("_", " ").capitalize()
                st.markdown(f"- {label}: {'✅' if v else '❌'}")

        if st.button("✅ Apply Profile"):
            tgt_grade = profile["grade_level"]
            simplify = profile["simplify_vocab"]
            short_p = profile["short_paragraphs"]
            define = profile["in_text_definitions"]
            breaks = profile["breaks"]
            make_qs = profile["make_qs"]
            st.success("Profile settings applied! Head to the 'Adapt Text' tab to continue.")
    else:
        st.info("Select a student profile to view and apply settings.")


# ──────────────────────────────────────  FOOTER  ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:13px;color:#86868b;'>Built with love • Powered by OpenAI</div>",
    unsafe_allow_html=True,
)
