import json
import os
import re
import tempfile
from pathlib import Path

import anthropic
import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import whisper
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ことばの力のものさし・デジタル版", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS: light modern gradient design
# ---------------------------------------------------------------------------
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Zen+Maru+Gothic:wght@400;700;900&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown("""<style>
/* ---------- global ---------- */
html, body, [class*="css"] {
    font-family: 'Zen Maru Gothic', 'Noto Sans JP', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 30%, #f5e6ff 60%, #ffecd2 100%);
    background-attachment: fixed;
}

/* Streamlit default overrides for light theme */
.stApp [data-testid="stHeader"] { background: transparent; }
.stApp [data-testid="stSidebar"] { background: rgba(255,255,255,0.7); }
h1, h2, h3, h4, h5, h6, p, li, span, label, div { color: #1e293b; }

/* ---------- hero ---------- */
.hero {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6366f1, #ec4899, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: 0.03em;
}
.hero p {
    color: #64748b;
    font-size: 1.05rem;
    font-weight: 400;
}

/* ---------- glass card ---------- */
.glass {
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 20px;
    backdrop-filter: blur(20px) saturate(1.8);
    -webkit-backdrop-filter: blur(20px) saturate(1.8);
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(99,102,241,0.08), 0 1px 4px rgba(0,0,0,0.04);
}

/* ---------- stage / step badge ---------- */
.stage-badge {
    display: inline-block;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.step-badge {
    display: inline-block;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ec4899, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.badge-label {
    color: #64748b;
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    letter-spacing: 0.05em;
}
.badge-sub {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}

/* ---------- feedback card ---------- */
.feedback-card {
    background: rgba(255,255,255,0.5);
    border-left: 4px solid;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    color: #334155;
    font-size: 0.93rem;
    line-height: 1.8;
    margin-top: 0.6rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.feedback-card.strengths    { border-color: #22c55e; background: rgba(34,197,94,0.06); }
.feedback-card.goals        { border-color: #f59e0b; background: rgba(245,158,11,0.06); }
.feedback-card.support      { border-color: #8b5cf6; background: rgba(139,92,246,0.06); }
.feedback-card.stage        { border-color: #6366f1; background: rgba(99,102,241,0.06); }
.feedback-card.step         { border-color: #ec4899; background: rgba(236,72,153,0.06); }
.feedback-card.comm         { border-color: #3b82f6; background: rgba(59,130,246,0.06); }
.feedback-card.codesw       { border-color: #a855f7; background: rgba(168,85,247,0.06); }
.feedback-card.interact     { border-color: #10b981; background: rgba(16,185,129,0.06); }
.feedback-card.creativity   { border-color: #f43f5e; background: rgba(244,63,94,0.06); }
.feedback-card.confidence   { border-color: #eab308; background: rgba(234,179,8,0.06); }
.feedback-card.selfrepair   { border-color: #14b8a6; background: rgba(20,184,166,0.06); }
.feedback-card.engagement   { border-color: #f97316; background: rgba(249,115,22,0.06); }

/* ---------- stage indicator ---------- */
.stage-bar {
    display: flex;
    gap: 6px;
    margin: 0.8rem 0;
}
.stage-bar .cell {
    flex: 1;
    text-align: center;
    padding: 0.6rem 0.2rem;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85rem;
    color: #94a3b8;
    background: rgba(255,255,255,0.5);
    border: 2px solid rgba(99,102,241,0.12);
    transition: all 0.3s ease;
}
.stage-bar .cell.active {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border: 2px solid #6366f1;
    color: #ffffff;
    box-shadow: 0 4px 16px rgba(99,102,241,0.3);
    transform: scale(1.05);
}

</style>""", unsafe_allow_html=True)

st.markdown("""<style>
/* ---------- step indicator ---------- */
.step-bar {
    display: flex;
    gap: 6px;
    margin: 0.8rem 0;
}
.step-bar .cell {
    flex: 1;
    text-align: center;
    padding: 0.6rem 0.2rem;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85rem;
    color: #94a3b8;
    background: rgba(255,255,255,0.5);
    border: 2px solid rgba(236,72,153,0.12);
    transition: all 0.3s ease;
}
.step-bar .cell.active {
    background: linear-gradient(135deg, #ec4899, #f59e0b);
    border: 2px solid #ec4899;
    color: #ffffff;
    box-shadow: 0 4px 16px rgba(236,72,153,0.3);
    transform: scale(1.05);
}

/* ---------- section headers ---------- */
h2 {
    color: #1e293b !important;
    font-weight: 900 !important;
    letter-spacing: 0.02em;
}
h3 {
    color: #334155 !important;
    font-weight: 700 !important;
}

/* ---------- misc tweaks ---------- */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: 16px;
    backdrop-filter: blur(12px);
}
[data-testid="stExpander"] p { color: #475569; }

/* metric value styling */
[data-testid="stMetricValue"] { color: #1e293b !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; }

/* file uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.5);
    border-radius: 16px;
    padding: 0.5rem;
}

/* selectbox */
[data-testid="stSelectbox"] label { color: #475569 !important; font-weight: 700; }

/* progress/status */
[data-testid="stStatusWidget"] {
    background: rgba(255,255,255,0.6) !important;
    border-radius: 16px !important;
}

/* ---------- about dialog ---------- */
.arch-flow {
    display: flex;
    align-items: stretch;
    gap: 0;
    margin: 1.2rem 0;
}
.arch-node {
    flex: 1;
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
    position: relative;
}
.arch-node h4 { margin: 0 0 0.5rem; font-size: 0.95rem; }
.arch-node ul { text-align: left; font-size: 0.82rem; line-height: 1.6; padding-left: 1.2rem; margin: 0; }
.arch-node.input   { background: linear-gradient(135deg, #dbeafe, #ede9fe); border: 2px solid #818cf8; }
.arch-node.audio   { background: linear-gradient(135deg, #ede9fe, #f3e8ff); border: 2px solid #a78bfa; }
.arch-node.whisper  { background: linear-gradient(135deg, #fef3c7, #ffedd5); border: 2px solid #f59e0b; }
.arch-node.llm     { background: linear-gradient(135deg, #fce7f3, #fdf2f8); border: 2px solid #ec4899; }
.arch-node.output  { background: linear-gradient(135deg, #d1fae5, #ecfdf5); border: 2px solid #34d399; }
.arch-arrow {
    display: flex;
    align-items: center;
    font-size: 1.5rem;
    color: #94a3b8;
    padding: 0 0.3rem;
}
.usage-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin: 0.8rem 0;
    padding: 1rem;
    background: rgba(99,102,241,0.05);
    border-radius: 14px;
    border-left: 4px solid #6366f1;
}
.usage-step .num {
    font-size: 1.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    min-width: 2rem;
}
.usage-step .desc { font-size: 0.93rem; color: #334155; line-height: 1.7; }

/* sidebar about button */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(12px);
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# About dialog
# ---------------------------------------------------------------------------

@st.dialog("このアプリについて", width="large")
def show_about():
    st.markdown("## ことばの力のものさし・デジタル版")
    st.markdown(
        "文部科学省「ことばの力のものさし」（2025年）に基づき、"
        "児童生徒の発話音声を AI で多角的に評価するアプリです。"
    )

    st.markdown("### 📐 処理の流れ（アーキテクチャ）")
    st.markdown("""
<div class="arch-flow">
    <div class="arch-node input">
        <h4>🎤 入力</h4>
        <ul>
            <li>音声ファイル<br>(.wav/.mp3/.m4a)</li>
            <li>学年段階の選択</li>
        </ul>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node audio">
        <h4>🔊 音声信号処理<br><small>Librosa</small></h4>
        <ul>
            <li>流暢さ（発話時間比率）</li>
            <li>ポーズ（頻度・長さ）</li>
            <li>韻律（ピッチ分析）</li>
            <li>エンゲージメント<br>（RMS + F0）</li>
        </ul>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node whisper">
        <h4>📝 音声認識<br><small>Whisper</small></h4>
        <ul>
            <li>日本語テキスト書き起こし</li>
            <li>発音自信度<br>（トークン確信度）</li>
            <li>フィラー・自己修復の検出</li>
            <li>発話速度の算出</li>
        </ul>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node llm">
        <h4>🤖 LLM 評価<br><small>Claude Sonnet 4</small></h4>
        <ul>
            <li>発達ステージ判定（A〜F）</li>
            <li>習得ステップ判定（1〜8）</li>
            <li>コミュニケーション方略</li>
            <li>コードスイッチング</li>
            <li>相互行為能力</li>
            <li>言語的創造性</li>
        </ul>
    </div>
    <div class="arch-arrow">→</div>
    <div class="arch-node output">
        <h4>📊 結果表示<br><small>Streamlit</small></h4>
        <ul>
            <li>ステージ・ステップ判定</li>
            <li>強み / 目標 / 支援提案</li>
            <li>音声・テキスト分析</li>
            <li>レーダーチャート</li>
        </ul>
    </div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("### 📖 使い方")
    st.markdown("""
<div class="usage-step">
    <div class="num">1</div>
    <div class="desc"><b>学年段階を選択</b><br>対象の児童生徒の学年段階（小1〜小2 / 小3〜小4 / 小5〜中2 / 中3〜高校）を選んでください。習得ステップの判定基準が変わります。</div>
</div>
<div class="usage-step">
    <div class="num">2</div>
    <div class="desc"><b>音声ファイルをアップロード</b><br>児童生徒の発話を録音した音声ファイル（.wav / .mp3 / .m4a）をドラッグ＆ドロップまたは選択してアップロードします。</div>
</div>
<div class="usage-step">
    <div class="num">3</div>
    <div class="desc"><b>自動で解析開始</b><br>アップロード後、自動的に音声分析 → 書き起こし → AI 評価が実行されます。数十秒〜1分程度お待ちください。</div>
</div>
<div class="usage-step">
    <div class="num">4</div>
    <div class="desc"><b>結果を確認</b><br>「発達ステージ（A〜F）」と「習得ステップ（1〜8）」の判定結果、強み・次の目標・支援のアドバイスが表示されます。音声分析の詳細データやテキスト分析も確認できます。</div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("### 📏 評価の2つの軸")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**包括的なことばの発達ステージ（A〜F）**

思考・判断・表現を支える力を評価します。

| ステージ | 名称 | 特徴 |
|:---:|:---|:---|
| A | イマココ期 | 断片的に話せる |
| B | イマココから順序期 | おおまかに順序立てて話せる |
| C | 順序期 | くわしく順序立てて話せる |
| D | 因果期 | 因果関係を含めて説明できる |
| E | 抽象期 | 抽象的概念を議論できる |
| F | 評価・発展期 | 批判的視点で議論できる |
        """)
    with col2:
        st.markdown("""
**日本語の習得ステップ（1〜8）**

日本語固有の知識・技能の習得状況を評価します。

- 学年段階ごとに異なる基準で判定
- ステップの進み具合は個人差が大きい
- 「聞く・話す」の技能に焦点

ステップ 1（初期段階）〜 ステップ 8（高度な運用）まで、段階的な成長を捉えます。
        """)

    st.markdown("---")
    st.markdown(
        "<small>基準: 文部科学省「ことばの発達と習得のものさし まるわかりガイド」（2025年4月）</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar: about button
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    if st.button("ℹ️ このアプリについて", use_container_width=True):
        show_about()


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>ことばの力のものさし・デジタル版</h1>
    <p>「ことばの力のものさし」に基づき、児童生徒の発話を AI で多角的に評価します</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input controls
# ---------------------------------------------------------------------------
GRADE_OPTIONS = ["小1〜小2段階", "小3〜小4段階", "小5〜中2段階", "中3〜高校段階"]

# Session state for sample mode
if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

input_col1, input_col2 = st.columns([2, 1])

with input_col1:
    uploaded_file = st.file_uploader(
        "音声ファイルをアップロード (.wav / .mp3 / .m4a)",
        type=["wav", "mp3", "m4a"],
    )

with input_col2:
    default_grade_idx = GRADE_OPTIONS.index("中3〜高校段階") if st.session_state.use_sample else 0
    grade_level = st.selectbox(
        "学年段階を選択",
        GRADE_OPTIONS,
        index=default_grade_idx,
    )

# Sample load button
sample_path = Path(__file__).parent / "sample.m4a"
sample_results_path = Path(__file__).parent / "sample_results.json"
if sample_path.exists() and uploaded_file is None and not st.session_state.use_sample:
    if st.button("🎧 サンプル音声で試してみる"):
        st.session_state.use_sample = True
        st.rerun()

# Determine audio source
if uploaded_file is not None:
    st.session_state.use_sample = False
    st.audio(uploaded_file)
    audio_source = uploaded_file
elif st.session_state.use_sample and sample_path.exists():
    st.audio(str(sample_path), format="audio/m4a")
    st.info("📎 サンプル音声の分析結果を表示しています")
    audio_source = "sample_precomputed"
else:
    audio_source = None

MIN_DURATION_SEC = 0.5

# ---------------------------------------------------------------------------
# Stage descriptions (包括的なことばの発達ステージ)
# ---------------------------------------------------------------------------
STAGE_INFO = {
    "A": "イマココ期",
    "B": "イマココから順序期",
    "C": "順序期",
    "D": "因果期",
    "E": "抽象期",
    "F": "評価・発展期",
}

# ---------------------------------------------------------------------------
# Fluency analysis
# ---------------------------------------------------------------------------

def analyze_fluency(audio_path: str) -> dict:
    """音声の無音/有音区間から流暢さ指標を算出する。"""
    y, sr = librosa.load(audio_path, sr=None)

    total_duration = len(y) / sr
    if total_duration < MIN_DURATION_SEC:
        raise ValueError(
            f"音声が短すぎます（{total_duration:.2f} 秒）。"
            f"{MIN_DURATION_SEC} 秒以上の音声をアップロードしてください。"
        )

    intervals = librosa.effects.split(y, top_db=40)
    speaking_duration = sum((end - start) for start, end in intervals) / sr
    silence_duration = total_duration - speaking_duration
    phonation_time_ratio = speaking_duration / total_duration if total_duration > 0 else 0.0

    return {
        "y": y,
        "sr": sr,
        "intervals": intervals,
        "total_duration": round(total_duration, 2),
        "speaking_duration": round(speaking_duration, 2),
        "silence_duration": round(silence_duration, 2),
        "phonation_time_ratio": round(phonation_time_ratio, 3),
    }


# ---------------------------------------------------------------------------
# Pause pattern analysis
# ---------------------------------------------------------------------------

def analyze_pause(intervals: np.ndarray, sr: int, total_duration: float) -> dict:
    """ポーズ（沈黙区間）の頻度と長さを分析する。"""
    pause_durations = []
    for i in range(1, len(intervals)):
        gap = (intervals[i][0] - intervals[i - 1][1]) / sr
        if gap > 0.1:
            pause_durations.append(gap)

    pause_count = len(pause_durations)
    mean_pause = float(np.mean(pause_durations)) if pause_durations else 0.0
    pause_rate = pause_count / total_duration * 60 if total_duration > 0 else 0.0

    return {
        "pause_count": pause_count,
        "mean_pause_duration": round(mean_pause, 3),
        "pause_rate_per_min": round(pause_rate, 1),
    }


# ---------------------------------------------------------------------------
# Speech rate analysis
# ---------------------------------------------------------------------------

def analyze_speech_rate(transcript: str, speaking_duration: float) -> dict:
    """発話速度を文字数/秒で算出する。"""
    char_count = len(transcript)
    rate = char_count / speaking_duration if speaking_duration > 0 else 0.0

    return {
        "char_count": char_count,
        "chars_per_sec": round(rate, 2),
    }


# ---------------------------------------------------------------------------
# Prosody (intonation) analysis
# ---------------------------------------------------------------------------

def analyze_prosody(y: np.ndarray, sr: int) -> dict:
    """ピッチの変動からイントネーションの豊かさを測定する。"""
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr,
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    if len(f0_valid) < 2:
        return {
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "pitch_range_hz": 0.0,
        }

    return {
        "pitch_mean_hz": round(float(np.mean(f0_valid)), 1),
        "pitch_std_hz": round(float(np.std(f0_valid)), 1),
        "pitch_range_hz": round(float(np.max(f0_valid) - np.min(f0_valid)), 1),
    }


# ---------------------------------------------------------------------------
# Transcription (Whisper)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


def transcribe_audio(audio_path: str) -> dict:
    """音声を書き起こし、トークンごとの確信度情報も返す。"""
    model = load_whisper_model()
    result = model.transcribe(
        audio_path,
        language="ja",
        word_timestamps=True,
    )
    text = result["text"].strip()
    if not text:
        raise ValueError("文字起こし結果が空です。音声に発話が含まれているか確認してください。")

    # セグメントからトークンごとの確信度を収集
    token_probs = []
    for seg in result.get("segments", []):
        for word_info in seg.get("words", []):
            token_probs.append({
                "word": word_info.get("word", ""),
                "probability": word_info.get("probability", 0.0),
            })

    return {"text": text, "segments": result.get("segments", []), "token_probs": token_probs}


# ---------------------------------------------------------------------------
# Pronunciation confidence analysis (Whisper logprob)
# ---------------------------------------------------------------------------

FILLER_PATTERNS = re.compile(
    r"(えーっと|えーと|えー|えっと|あのー|あの|うーん|うん|んー|まあ|ええと|そのー|その|なんか)"
)


def analyze_pronunciation_confidence(token_probs: list[dict]) -> dict:
    """Whisper のトークン確信度から発音の自信度を分析する。"""
    if not token_probs:
        return {
            "mean_confidence": 0.0,
            "low_confidence_words": [],
            "confidence_score": 0.0,
        }

    probs = [tp["probability"] for tp in token_probs]
    mean_conf = float(np.mean(probs))

    # 確信度が低いワード（0.5未満）を抽出
    low_conf = [
        {"word": tp["word"].strip(), "confidence": round(tp["probability"], 3)}
        for tp in token_probs
        if tp["probability"] < 0.5 and tp["word"].strip()
    ]

    return {
        "mean_confidence": round(mean_conf, 3),
        "low_confidence_words": low_conf[:10],  # 上位10件
        "confidence_score": round(mean_conf * 100, 1),
    }


# ---------------------------------------------------------------------------
# Self-repair & hesitation analysis
# ---------------------------------------------------------------------------

def analyze_self_repair(transcript: str, speaking_duration: float) -> dict:
    """フィラー・言い直しの頻度を分析する。"""
    fillers = FILLER_PATTERNS.findall(transcript)
    filler_count = len(fillers)
    filler_rate = filler_count / speaking_duration * 60 if speaking_duration > 0 else 0.0

    return {
        "filler_count": filler_count,
        "filler_examples": list(set(fillers))[:5],
        "filler_rate_per_min": round(filler_rate, 1),
    }


# ---------------------------------------------------------------------------
# Engagement analysis (RMS energy + F0 dynamics)
# ---------------------------------------------------------------------------

def analyze_engagement(y: np.ndarray, sr: int, prosody: dict) -> dict:
    """声の張り（RMSエネルギー）とピッチ変動からエンゲージメントを推定する。"""
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))
    std_rms = float(np.std(rms))
    rms_cv = std_rms / mean_rms if mean_rms > 0 else 0

    # エンゲージメント指標: 声量の変化 + ピッチの変動
    pitch_mean = prosody["pitch_mean_hz"]
    pitch_std = prosody["pitch_std_hz"]
    pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0

    # 高エンゲージメント = 声量にメリハリ + ピッチに抑揚
    energy_score = np.clip(rms_cv / 0.5 * 50, 0, 50)
    pitch_score = np.clip(pitch_cv / 0.30 * 50, 0, 50)
    engagement_score = round(float(energy_score + pitch_score), 1)

    # 声量が低く平坦な場合は不安・自信のなさの可能性
    is_flat = rms_cv < 0.15 and pitch_cv < 0.10

    return {
        "mean_rms": round(mean_rms, 4),
        "rms_cv": round(rms_cv, 3),
        "engagement_score": engagement_score,
        "is_flat": is_flat,
    }


# ---------------------------------------------------------------------------
# Step descriptors per grade level (聞く・話す)
# ---------------------------------------------------------------------------

STEP_RUBRIC = {
    "小1〜小2段階": {
        1: "よく耳にする単語やその一部を口にする。質問に答えられず沈黙したり、おうむ返しする場合がある。指示の意味が理解できなくても反応する（うなずく等）。",
        2: "ゆっくりはっきりした質問に限られた単語で答えられる。定型表現（「おはよう」等）を使える。日常生活で簡単な質問ができる。覚えたばかりの決まった形でやりとりができる。",
        3: "対話による支援を得て、よく耳にする語彙・表現を使って主に単文でなんとか意味を通じさせる。日常生活で教師や友達に働きかけるために必要最低限のやりとりができる。",
        4: "対話による支援を得て、よく耳にする語彙・表現を使い、単文や簡単な複文で話せる。日常生活で教師や友達に働きかけるために必要最低限のやりとりができる。",
        5: "身近な場面や関心のある話題について、日常的な語彙・表現を幅広く使って対話・自発的発話ができる。接続表現を活用し、ほぼ頻りなく自由に話を続けられる。「です・ます」が使える。",
    },
    "小3〜小4段階": {
        1: "自分の名前や学年・組など自分に関係のある語がおおむねわかる。支援者と一緒に1文字ずつ拾い読みする。",
        2: "ゆっくりはっきりした質問に限られた単語で答える。定型表現を使って日直などの係の司会ができる。簡単な質問ができる。",
        3: "対話による支援を得て、日常でよく耳にする語彙・表現を使い、単文や簡単な複文で話せる。教師や友達に働きかけるために必要最低限のやりとりができる。",
        4: "身近な場面や関心のある話題について、日常でよく耳にする語彙・表現を使い、単文や簡単な複文で話せる。場面に応じて必要な情報を含むやりとりができる。",
        5: "身近な場面や関心のある話題について、日常的な語彙・表現を幅広く使って話せる。まとまりのある話を自然な速度で聞き取ることができる。",
        6: "教科学習内容の基本的な概念の話（中学年レベル）を既習の基本的な語彙・表現を使って話せる。自然な速度で聞き取れる。接続表現を使いまとまり（結束性）がある話ができる。",
        7: "高学年から中学レベルの既習の慣用的な表現・コロケーションが増え適切に使える。場面に応じた敬語表現が使える。教科学習内容の抽象的な概念の話を既習の語彙・表現で話せる。",
    },
    "小5〜中2段階": {
        1: "自分の名前や学年・組など自分に関係のある語がおおむねわかる。連絡帳や時間割などで毎日使うマークがわかる。",
        2: "ゆっくりはっきりした質問に限られた単語で答える。よく使われる定型表現を使える。簡単な質問ができる。",
        3: "身近な場面や関心のある話題について、対話による支援を得て基本的な語彙・表現を使い、単文や簡単な複文で話せる。",
        4: "身近な場面や関心のある話題について、既習の語彙・表現・文型を使い、単文や簡単な複文で話せる。必要な情報を含むやりとりができる。理由を伝えて断るなどができる。",
        5: "日常的な語彙・表現や接続表現を幅広く使い、ほぼ頻りなく自由に話を続けられる。まとまりのある話を自然な速度で聞き取ることができる。",
        6: "教科学習内容の抽象的な概念の話（高学年〜中学レベル）を既習の基本的な概念語彙・表現で話せる。自然な速度で聞き取れる。",
        7: "高学年から中学レベルの既習の慣用的な表現・コロケーションが増え適切に使える。場面に応じた敬語表現を使える。話体を選択して話せる。",
        8: "教科学習内容の抽象的な概念の話など（中学・高校レベル）を既習の抽象的な概念語彙・表現を幅広く使って話せる。このような話を自然な速度で聞き取ることができる。",
    },
    "中3〜高校段階": {
        1: "自分の名前や学年・組・学校名など自分に関係のある語がおおむねわかる。毎日使うマークがわかる。",
        2: "ゆっくりはっきりした質問に限られた単語で答える。定型表現を使える。日常生活で簡単な質問ができる。",
        3: "身近な場面や関心のある話題について、基本的な語彙・表現を使い、単文を連ねて文章が書ける。学校生活で必要最低限のやりとりができる。",
        4: "身近な場面や関心のある話題について、既習の語彙・表現を使い、単文や簡単な複文で話せる。必要な情報を含むやりとりができる。",
        5: "日常的な語彙・表現を幅広く使い、ほぼ頻りなく自由に話を続けられる。まとまりのある話を自然な速度で聞き取ることができる。",
        6: "教科学習内容の抽象的な概念の話を既習の基本的な概念語彙・表現で話せる。自然な速度で聞き取ることができる。",
        7: "高学年から中学レベルの既習の慣用的な表現・コロケーションが増え適切に使える。場面に応じた敬語表現を使える。教科学習内容の抽象的な概念の話を既習の語彙・表現で話せる。",
        8: "中学から高校レベルの教科学習に必要な抽象的な概念語彙・表現を幅広く使って話せる。このような話を自然な速度で聞き取ることができる。",
    },
}


# ---------------------------------------------------------------------------
# Build system prompt dynamically based on grade level
# ---------------------------------------------------------------------------

def build_system_prompt(grade: str) -> str:
    rubric = STEP_RUBRIC[grade]
    step_lines = "\n".join(
        f"- ステップ{num}: {desc}" for num, desc in sorted(rubric.items())
    )

    return f"""\
あなたは「ことばの力のものさし」（文部科学省, 2025年）に基づいて、\
外国人児童生徒の日本語の発話を評価する専門家です。

音声から書き起こされた発話テキストが与えられます。
対象の学年段階は「{grade}」です。
「聞く・話す」の観点から、以下の2つの軸で評価してください。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 軸1: 包括的なことばの発達ステージ（A〜F）
思考・判断・表現を支える包括的なことばの力（複数言語での力）を「聞く・話す」の観点から判定します。
日本語も母語も含めて子どもが持っているすべてのことばのレパートリーを使って最大限にできることを評価します。

- ステージA【イマココ期】: 対話による支援を得て、身近なことや経験したことについて覚えている場面を断片的に話せる。ごく簡単な質問（誰が、何が/を、どんな/どうした等）に答えられる。
- ステージB【イマココから順序期】: 対話による支援を得て、身近なことや経験したことについて順序にそっておおまかに話せる。学習内容を聞いておおむね理解し、ひとこと程度の感想が言える。自分が聞きたいことを質問できる。
- ステージC【順序期】: 自分に関係のあることや体験したことについて順序にそってくわしく話せる。学習内容を聞いて話の流れを理解し、感想とその理由が言える。話し合いの場で教師や友達の話を聞いて発言できる。
- ステージD【因果期】: 教科学習内容の基本的な概念について因果関係を含めて説明できる。集めた情報を示しながら授業で発表できる。具体的な事例とともに理由を挙げながら自分の意見を述べられる。
- ステージE【抽象期】: 抽象的な概念について事実と意見の違いを意識しつつ、共通点や相違点を整理して議論できる。構成を意識したわかりやすいプレゼンテーションができる。場面や相手に応じて適切な語彙や表現を選択できる。論拠を示しながらおおむね一貫性のある意見を述べられる。
- ステージF【評価・発展期】: 中学から高校の教科学習内容について多角的・批判的視点をもった議論ができる。論理的構成を意識し根拠に基づいた効果的なプレゼンテーションができる。反論できる論理の展開を考え説得力のある意見を述べられる。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 軸2: 日本語の習得ステップ（1〜8）〈聞く・話す〉
日本語固有の知識・技能の習得状況を「{grade}」の基準で判定します。
ステップの進み具合は個人差が大きく、数ヶ月でいくつものステップを進めるケースもあれば、数年同じステップにとどまるケースもあります。

{step_lines}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 追加の分析観点（テキストから判定）

以下の4つの観点についても分析してください。

3. **コミュニケーション方略（Communication Strategies）**: 知らない単語があったとき、黙り込むのではなく知っている言葉で言い換え（Circumlocution）をして伝えようとする力。対象年齢の標準的な語彙を迂回して説明している箇所を抽出し、言語的なサバイバル能力を評価する。
4. **言語間移動・コードスイッチング（Translanguaging / Code-switching）**: L2の語彙が不足した際にとっさに母語（L1）を交えて発話する現象。L2の文脈にL1の単語がどのように挿入されているかを判定する。これをエラーとしてではなく、多言語リソースの活用として評価する。
5. **相互行為能力（Interactive Competence）**: 相手の問いかけに対する応答の適切さ、「うん」「へえ」などの相槌（バックチャネル）の使用、会話の流れへの参加度合いを評価する（対話形式データの場合）。
6. **言語的創造性（Linguistic Creativity）**: 擬音語・擬態語（オノマトペ）の多用、子ども特有の言葉遊び、未知語の自作など、正誤判定だけでは捉えられない表現力や場面を生き生きと伝える力を評価する。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 出力フォーマット（厳守）

挨拶・前置き・説明文などは一切含めず、純粋な JSON だけを出力してください。

```json
{{
  "stage": "<A〜Fのいずれか>",
  "stage_reasoning": "<ステージ判定の根拠を具体的に。発話のどの部分がどのステージの特徴に該当するか（日本語、2〜4文）>",
  "step": <1〜8の整数>,
  "step_reasoning": "<ステップ判定の根拠を具体的に。発話のどの部分がどのステップの特徴に該当するか（日本語、2〜4文）>",
  "communication_strategies": "<コミュニケーション方略の分析。言い換えや迂回表現が見られた箇所とその評価（日本語、1〜3文）。該当なしの場合はその旨>",
  "code_switching": "<コードスイッチングの分析。L1混入の箇所とその文脈での役割（日本語、1〜3文）。該当なしの場合はその旨>",
  "interactive_competence": "<相互行為能力の分析。応答の適切さ、相槌の使用など（日本語、1〜3文）。対話でない場合はその旨>",
  "linguistic_creativity": "<言語的創造性の分析。オノマトペ、言葉遊び、創造的表現の使用とその評価（日本語、1〜3文）。該当なしの場合はその旨>",
  "strengths": "<この発話で見られる強み・できていること（日本語、2〜3文）>",
  "next_goals": "<次の段階に進むための目標・改善点（日本語、2〜3文）>",
  "support_suggestions": "<指導者・支援者へのアドバイス。この子どもに有効な支援方法の提案（日本語、2〜3文）>"
}}
```"""


# ---------------------------------------------------------------------------
# LLM evaluation (Claude API)
# ---------------------------------------------------------------------------

def analyze_text_with_llm(transcript: str, grade: str) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY が設定されていません。`.env` ファイルを確認してください。"
        )

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = build_system_prompt(grade)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": transcript}],
        )
    except anthropic.RateLimitError:
        raise RuntimeError("Anthropic API のレート制限に達しました。しばらく待ってから再試行してください。")
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API エラー: {e}")

    raw = message.content[0].text
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    text_to_parse = m.group(1).strip() if m else raw.strip()

    try:
        return json.loads(text_to_parse)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"LLM の応答を JSON としてパースできませんでした。\n\n応答内容:\n{raw}"
        )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def draw_stage_bar(current_stage: str) -> str:
    """ステージ A〜F のインジケーター HTML を生成する。"""
    cells = []
    for s, name in STAGE_INFO.items():
        active = "active" if s == current_stage else ""
        cells.append(f'<div class="cell {active}">{s}<br><small>{name}</small></div>')
    return f'<div class="stage-bar">{"".join(cells)}</div>'


def draw_step_bar(current_step: int, max_step: int) -> str:
    """ステップ 1〜max_step のインジケーター HTML を生成する。"""
    cells = []
    for i in range(1, max_step + 1):
        active = "active" if i == current_step else ""
        cells.append(f'<div class="cell {active}">{i}</div>')
    return f'<div class="step-bar">{"".join(cells)}</div>'


def draw_audio_radar(fluency: dict, pause: dict, speech_rate: dict, prosody: dict) -> go.Figure:
    """音声指標のレーダーチャートを描画する。"""
    # 各指標を 0-100 に正規化
    fluency_score = np.clip((fluency["phonation_time_ratio"] - 0.2) / 0.7 * 100, 0, 100)

    mean_pause = pause["mean_pause_duration"]
    pause_rate = pause["pause_rate_per_min"]
    length_pen = np.clip((mean_pause - 0.3) / 1.5 * 50, 0, 50) if mean_pause > 0.3 else 0
    freq_pen = np.clip((pause_rate - 10) / 20 * 50, 0, 50) if pause_rate > 10 else 0
    pause_score = np.clip(100 - length_pen - freq_pen, 0, 100)

    rate_score = np.clip(speech_rate["chars_per_sec"] / 6.0 * 100, 0, 100)

    pitch_mean = prosody["pitch_mean_hz"]
    pitch_std = prosody["pitch_std_hz"]
    cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
    prosody_score = np.clip(cv / 0.30 * 100, 0, 100)

    categories = ["流暢さ", "ポーズ", "発話速度", "韻律"]
    values = [
        round(float(fluency_score), 1),
        round(float(pause_score), 1),
        round(float(rate_score), 1),
        round(float(prosody_score), 1),
    ]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=categories_closed, fill="toself",
        line=dict(color="rgba(99,102,241,0.5)", width=3),
        fillcolor="rgba(99,102,241,0.08)",
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=categories_closed, fill="toself",
        name="音声指標",
        line=dict(color="#6366f1", width=2),
        fillcolor="rgba(139,92,246,0.15)",
        marker=dict(size=8, color="#ec4899"),
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                tickfont=dict(color="#94a3b8", size=10),
                gridcolor="rgba(99,102,241,0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(color="#475569", size=13, family="Zen Maru Gothic, Noto Sans JP"),
                gridcolor="rgba(99,102,241,0.1)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=40, b=40, l=60, r=60),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Main: display results
# ---------------------------------------------------------------------------

if audio_source is not None:
    # ---- Sample precomputed mode ----
    if audio_source == "sample_precomputed" and sample_results_path.exists():
        with open(sample_results_path, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        fluency = sample_data["fluency"]
        pause = sample_data["pause"]
        prosody = sample_data["prosody"]
        speech_rate = sample_data["speech_rate"]
        pronunciation = sample_data["pronunciation"]
        self_repair = sample_data["self_repair"]
        engagement = sample_data["engagement"]
        llm_result = sample_data["llm_result"]
        transcript = sample_data["transcript"]
        _is_temp = False

    # ---- Normal upload mode ----
    else:
        suffix = Path(audio_source.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_source.getvalue())
            tmp_path = tmp.name
        _is_temp = True

        with st.status("解析を実行中...", expanded=True) as status:
            st.write("🔊 音声を解析中...")
            try:
                fluency_raw = analyze_fluency(tmp_path)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            st.write("🔍 ポーズ・韻律を分析中...")
            pause = analyze_pause(fluency_raw["intervals"], fluency_raw["sr"], fluency_raw["total_duration"])
            prosody = analyze_prosody(fluency_raw["y"], fluency_raw["sr"])

            st.write("📝 音声をテキスト化中...")
            try:
                whisper_result = transcribe_audio(tmp_path)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            transcript = whisper_result["text"]
            token_probs = whisper_result["token_probs"]

            st.write("🔬 発音自信度・自己修復・エンゲージメントを分析中...")
            speech_rate = analyze_speech_rate(transcript, fluency_raw["speaking_duration"])
            pronunciation = analyze_pronunciation_confidence(token_probs)
            self_repair = analyze_self_repair(transcript, fluency_raw["speaking_duration"])
            engagement = analyze_engagement(fluency_raw["y"], fluency_raw["sr"], prosody)

            st.write("🤖 ことばの力のものさしで評価中...")
            try:
                llm_result = analyze_text_with_llm(transcript, grade_level)
            except (EnvironmentError, RuntimeError) as e:
                st.error(str(e))
                st.stop()

            status.update(label="解析完了 ✨", state="complete", expanded=False)

            # Store serializable fluency data
            fluency = {k: v for k, v in fluency_raw.items() if k not in ("y", "sr", "intervals")}

        # Clean up temp file
        if _is_temp:
            Path(tmp_path).unlink(missing_ok=True)

    # ==================================================================
    # Results: Stage & Step
    # ==================================================================
    stage = llm_result.get("stage", "?")
    stage_name = STAGE_INFO.get(stage, "")
    step = llm_result.get("step", 0)
    max_step = max(STEP_RUBRIC[grade_level].keys())

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    badge_col, indicator_col = st.columns([1, 2])

    with badge_col:
        st.markdown(
            f'<div style="text-align:center;">'
            f'<div class="badge-label">発達ステージ</div>'
            f'<div class="stage-badge">ステージ {stage}</div>'
            f'<div class="badge-sub">【{stage_name}】</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;">'
            f'<div class="badge-label">習得ステップ（{grade_level}）</div>'
            f'<div class="step-badge">ステップ {step}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with indicator_col:
        st.markdown("#### 包括的なことばの発達ステージ（A〜F）")
        st.markdown(draw_stage_bar(stage), unsafe_allow_html=True)
        st.markdown(
            f'<div class="feedback-card stage">{llm_result.get("stage_reasoning", "")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"#### 日本語の習得ステップ（1〜{max_step}）")
        st.markdown(draw_step_bar(step, max_step), unsafe_allow_html=True)
        st.markdown(
            f'<div class="feedback-card step">{llm_result.get("step_reasoning", "")}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ==============================================================
    # Feedback: strengths / goals / support
    # ==============================================================
    fb_col1, fb_col2, fb_col3 = st.columns(3)

    with fb_col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 💪 強み・できていること")
        st.markdown(
            f'<div class="feedback-card strengths">{llm_result.get("strengths", "")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with fb_col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 🎯 次の目標")
        st.markdown(
            f'<div class="feedback-card goals">{llm_result.get("next_goals", "")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with fb_col3:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 📖 支援のアドバイス")
        st.markdown(
            f'<div class="feedback-card support">{llm_result.get("support_suggestions", "")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ==============================================================
    # Audio-based analysis: confidence, self-repair, engagement
    # ==============================================================
    st.markdown("## 🔬 音声・認識モデルの分析")
    au_col1, au_col2, au_col3 = st.columns(3)

    with au_col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 🎤 発音の自信度")
        conf_pct = f"{pronunciation['mean_confidence']:.1%}"
        st.metric("平均確信度", conf_pct)
        low_words = pronunciation["low_confidence_words"]
        if low_words:
            word_list = "、".join(
                f"**{w['word']}**（{w['confidence']:.0%}）" for w in low_words[:5]
            )
            st.markdown(
                f'<div class="feedback-card confidence">確信度が低い箇所: {word_list}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="feedback-card confidence">全体的に安定した発音です。</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with au_col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 🔄 自己修復・フィラー")
        st.metric("フィラー回数", f"{self_repair['filler_count']} 回")
        st.metric("フィラー頻度", f"{self_repair['filler_rate_per_min']} 回/分")
        examples = self_repair["filler_examples"]
        if examples:
            ex_str = "、".join(f"「{e}」" for e in examples)
            st.markdown(
                f'<div class="feedback-card selfrepair">検出されたフィラー: {ex_str}<br>'
                f'自己修復はメタ認知の現れでもあり、自分の発話を振り返れている証拠です。</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="feedback-card selfrepair">フィラーや言い直しは検出されませんでした。</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with au_col3:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 😊 エンゲージメント")
        st.metric("エンゲージメントスコア", f"{engagement['engagement_score']} / 100")
        if engagement["is_flat"]:
            msg = "声量・ピッチともに平坦な傾向があります。不安感や自信のなさが見られるかもしれません。安心して話せる環境づくりが大切です。"
        else:
            msg = "声の張りやピッチの変動があり、発話に対する意欲的な姿勢が感じられます。"
        st.markdown(
            f'<div class="feedback-card engagement">{msg}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ==============================================================
    # LLM-based extended analysis
    # ==============================================================
    st.markdown("## 🧠 テキスト分析（LLM）")
    lx_col1, lx_col2 = st.columns(2)

    with lx_col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 💡 コミュニケーション方略")
        st.markdown(
            f'<div class="feedback-card comm">{llm_result.get("communication_strategies", "該当なし")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 🌐 コードスイッチング")
        st.markdown(
            f'<div class="feedback-card codesw">{llm_result.get("code_switching", "該当なし")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with lx_col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 🤝 相互行為能力")
        st.markdown(
            f'<div class="feedback-card interact">{llm_result.get("interactive_competence", "該当なし")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### ✨ 言語的創造性")
        st.markdown(
            f'<div class="feedback-card creativity">{llm_result.get("linguistic_creativity", "該当なし")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ==============================================================
    # Audio metrics (collapsible)
    # ==============================================================
    with st.expander("🎙️ 音声分析の詳細データ"):
        chart_col, data_col = st.columns([3, 2])

        with chart_col:
            fig = draw_audio_radar(fluency, pause, speech_rate, prosody)
            st.plotly_chart(fig, use_container_width=True)

        with data_col:
            st.markdown("**発話時間**")
            st.metric("総時間", f"{fluency['total_duration']} 秒")
            st.metric("発話時間", f"{fluency['speaking_duration']} 秒")
            st.metric("発話比率", f"{fluency['phonation_time_ratio']:.1%}")

            st.markdown("**ポーズ**")
            st.metric("ポーズ回数", f"{pause['pause_count']} 回")
            st.metric("平均ポーズ長", f"{pause['mean_pause_duration']} 秒")

            st.markdown("**発話速度**")
            st.metric("速度", f"{speech_rate['chars_per_sec']} 字/秒")

            st.markdown("**韻律**")
            st.metric("平均ピッチ", f"{prosody['pitch_mean_hz']} Hz")
            st.metric("ピッチ範囲", f"{prosody['pitch_range_hz']} Hz")

    # ==============================================================
    # Transcript
    # ==============================================================
    with st.expander("💬 文字起こしテキストを表示"):
        st.write(transcript)

