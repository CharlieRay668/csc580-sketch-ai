
# pictionary app for csc 580
# run with: streamlit run app/app.py (from project root)

import streamlit as st
import numpy as np
import time
import random
from PIL import Image
from pathlib import Path
from streamlit_drawable_canvas import st_canvas
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# try loading our models, fall back to random guesses if torch isnt set up
MODELS_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    from src.models import create_model
    from src.data import ALL_QUICKDRAW_CATEGORIES
    from torchvision import transforms
    MODELS_AVAILABLE = True
except ImportError:
    ALL_QUICKDRAW_CATEGORIES = [
        'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel',
        'animal migration', 'ant', 'anvil', 'apple', 'arm',
        'asparagus', 'axe', 'backpack', 'banana', 'bandage',
        'barn', 'baseball', 'baseball bat', 'basket', 'basketball',
        'bat', 'bathtub', 'beach', 'bear', 'beard',
        'bed', 'bee', 'belt', 'bench', 'bicycle',
        'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry',
        'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet',
        'brain', 'bread', 'bridge', 'broccoli', 'broom',
        'bucket', 'bulldozer', 'bus', 'bush', 'butterfly',
    ]

# optional auto-refresh dependency, falls back to manual refresh button
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

st.set_page_config(page_title="Pictionary", page_icon="✏️", layout="wide",
                   initial_sidebar_state="collapsed")

ROUND_TIME = 30
TOTAL_ROUNDS = 5
CANVAS_SIZE = 400

# (display name, model key, checkpoint filename, accent color)
MODELS = [
    ("MLP",       "mlp",      "mlp_best.pth",      "#c0583a"),
    ("ResNet-18", "resnet18",  "resnet18_best.pth", "#2d6a4f"),
    ("ViT",       "vit",       "vit_best.pth",      "#3d5a80"),
]

MODEL_DESCS = {
    "MLP": "Multi-Layer Perceptron",
    "ResNet-18": "Residual Network",
    "ViT": "Vision Transformer",
}

# styling - mostly just overriding streamlit defaults to get the paper look
# we load two google fonts and set up some reusable classes
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&display=swap');

.stApp { background: #f5f0e8 !important; font-family: 'DM Sans', Georgia, serif; }
section[data-testid="stSidebar"] { background: #ece7dc; }
h1, h2, h3, h4, h5, h6, p, span, div, label,
.stMarkdown, .stText { color: #2c2c2c !important; }
#MainMenu, footer, header, .stDeployButton { display: none !important; }

.title-main {
    font-family: 'Libre Baskerville', Georgia, serif;
    font-size: 56px; font-weight: 700; text-align: center;
    line-height: 1.15; margin: 0; letter-spacing: -1px;
}
.title-sub {
    text-align: center; color: #6b6560 !important;
    font-size: 15px; margin-top: 6px; margin-bottom: 36px;
}
.label-sm {
    font-size: 11px; font-weight: 500; color: #b5aea5 !important;
    text-transform: uppercase; letter-spacing: 1.5px;
}
.paper-card {
    background: #fffdf8; border: 1px solid #ddd5c8;
    border-radius: 6px; padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.draw-this {
    font-family: 'Libre Baskerville', Georgia, serif;
    font-size: 36px; font-weight: 700; font-style: italic;
    text-transform: capitalize; line-height: 1.2;
}
.timer-num {
    font-family: 'Libre Baskerville', Georgia, serif;
    font-size: 48px; font-weight: 700; text-align: center; line-height: 1;
}
.timer-ok   { color: #2d6a4f !important; }
.timer-warn { color: #b8860b !important; }
.timer-crit { color: #c0583a !important; }
.score-pill {
    font-weight: 700; font-size: 14px;
    background: #fff3cd; padding: 5px 16px;
    border-radius: 20px; display: inline-block; border: 1px solid #e8dbb3;
}
.model-tag {
    font-size: 12px; font-weight: 700; padding: 3px 10px;
    border-radius: 4px; display: inline-block;
}
.correct-chip {
    font-size: 10px; font-weight: 700; color: #fff !important;
    background: #2d6a4f; padding: 2px 8px; border-radius: 3px;
    display: inline-block;
}
.conf-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.conf-track { flex: 1; background: #ddd5c8; border-radius: 4px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 4px; }
.result-row {
    display: flex; align-items: center; gap: 14px;
    padding: 14px 18px; background: #fffdf8;
    border: 1px solid #ddd5c8; border-radius: 6px; margin-bottom: 8px;
}
.result-row-hit  { border-left: 3px solid #2d6a4f; }
.result-row-miss { border-left: 3px solid #c0583a; }
.result-icon {
    width: 28px; height: 28px; border-radius: 50%;
    font-size: 13px; font-weight: 700; color: #fff !important;
    text-align: center; line-height: 28px; flex-shrink: 0;
}
.rule { border: none; border-top: 1px solid #ddd5c8; margin: 20px 0; }
.waiting-msg {
    text-align: center; padding: 40px 16px; color: #b5aea5 !important;
    font-family: 'Libre Baskerville', Georgia, serif;
    font-style: italic; font-size: 15px;
}
</style>
""", unsafe_allow_html=True)


# session state defaults
for key, default in {
    "game_state": "menu",
    "round": 0, "score": 0,
    "target_word": "",
    "round_start_time": None,
    "canvas_id": 0,
    "ai_results": [],
    "round_history": [],
    "used_words": [],
    "brush_size": 8,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource
def load_models(models_dir="models"):
    """loads all three checkpoints if they exist. cached so we only do this once."""
    if not MODELS_AVAILABLE:
        return None

    models_path = Path(models_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded = {}

    for name, mtype, fname, _ in MODELS:
        path = models_path / fname
        if not path.exists():
            continue
        try:
            ckpt = torch.load(path, map_location=device)
            sd = ckpt['model_state_dict']

            # each model type stores the output layer under a different key
            if 'head.weight' in sd:
                nc = sd['head.weight'].shape[0]
            elif 'fc.weight' in sd:
                nc = sd['fc.weight'].shape[0]
            elif 'mlp_head.1.weight' in sd:
                nc = sd['mlp_head.1.weight'].shape[0]
            else:
                nc = 50

            model = create_model(mtype, num_classes=nc)
            model.load_state_dict(sd)
            model.to(device).eval()
            loaded[name] = (model, nc, device)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")

    return loaded if loaded else None


def _inference_transform():
    # matches the normalization from training
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def run_inference(models_dict, pil_gray):
    """runs all three models on the grayscale drawing and returns top-5 guesses each"""
    tensor = _inference_transform()(pil_gray)
    results = []

    for name, mtype, _, color in MODELS:
        if name not in models_dict:
            continue
        model, nc, device = models_dict[name]
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0]
            vals, idxs = torch.topk(probs, k=min(5, nc))

        cats = ALL_QUICKDRAW_CATEGORIES[:nc]
        guesses = [{"label": cats[i.item()], "conf": v.item()} for v, i in zip(vals, idxs)]
        results.append({
            "model": name, "color": color, "guesses": guesses,
            "top_guess": guesses[0]["label"], "top_conf": guesses[0]["conf"],
        })
    return results


def simulate_inference(target):
    """placeholder guesses for when we dont have the checkpoint files yet.
    just picks 3 random categories per model and assigns random confidences.
    one of the 3 has a chance of being the correct answer so the game still works."""
    cats = ALL_QUICKDRAW_CATEGORIES[:50]
    results = []

    for name, _, _, color in MODELS:
        # give each model a small chance of guessing right
        pool = [c for c in cats if c != target]
        random.shuffle(pool)
        top3 = pool[:3]
        if random.random() < 0.35:
            top3[0] = target

        # random confidences, sorted so the highest is first
        confs = sorted([random.random() * 0.7 + 0.05 for _ in range(3)], reverse=True)
        guesses = [{"label": top3[i], "conf": confs[i]} for i in range(3)]

        results.append({
            "model": name, "color": color, "guesses": guesses,
            "top_guess": guesses[0]["label"], "top_conf": guesses[0]["conf"],
        })
    return results


def pick_word():
    """picks a random category we havent used yet this game"""
    pool = [c for c in ALL_QUICKDRAW_CATEGORIES[:50] if c not in st.session_state.used_words]
    if not pool:
        pool = ALL_QUICKDRAW_CATEGORIES[:50]
    return random.choice(pool)


def get_drawing(canvas_result):
    """extracts a grayscale pil image from the canvas. returns None if nothing drawn."""
    if canvas_result.image_data is None:
        return None
    arr = canvas_result.image_data.astype(np.uint8)
    if arr[:, :, 3].max() == 0:
        return None
    return Image.fromarray(arr, mode="RGBA").convert("L")


def infer_or_simulate(pil_gray, target):
    """runs real models if checkpoints are available, otherwise uses placeholder guesses"""
    models = load_models()
    if models:
        return run_inference(models, pil_gray)
    return simulate_inference(target)


def conf_bar(label, conf, color, top=False, correct=False):
    """renders one confidence bar row as html"""
    pct = round(conf * 100)
    lbl_color = "#2c2c2c" if top else "#b5aea5"
    fill = "#2d6a4f" if (top and correct) else color if top else "#b5aea5"
    h = 10 if top else 6
    w = 700 if top else 400
    op = "1" if top else "0.4"

    st.markdown(f"""
    <div class="conf-row">
        <span style="font-size:12px; width:95px; text-align:right; overflow:hidden;
              text-overflow:ellipsis; white-space:nowrap;
              color:{lbl_color}; font-weight:{w};">{label}</span>
        <div class="conf-track" style="height:{h}px;">
            <div class="conf-fill" style="width:{pct}%; background:{fill}; opacity:{op};"></div>
        </div>
        <span style="font-size:11px; width:32px; text-align:right;
              color:{lbl_color}; font-weight:{w};">{pct}%</span>
    </div>""", unsafe_allow_html=True)


def model_card(result, target):
    """renders the card for one model showing its top 3 guesses"""
    hit = result["top_guess"] == target
    border = "#2d6a4f" if hit else "#ddd5c8"
    badge = '<span class="correct-chip">CORRECT</span>' if hit else ""
    c = result["color"]

    st.markdown(f"""
    <div class="paper-card" style="border-color:{border}; margin-bottom:14px;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
            <span class="model-tag" style="background:{c}18; color:{c} !important;">{result['model']}</span>
            {badge}
        </div>
    </div>""", unsafe_allow_html=True)

    for i, g in enumerate(result["guesses"][:3]):
        conf_bar(g["label"], g["conf"], c,
                 top=(i == 0), correct=(i == 0 and g["label"] == target))


def start_round():
    """sets up state for a new round"""
    word = pick_word()
    st.session_state.target_word = word
    st.session_state.used_words.append(word)
    st.session_state.round_start_time = time.time()
    st.session_state.canvas_id += 1
    st.session_state.ai_results = []


def end_round(canvas_result=None):
    """finalizes the round - runs one last inference, calculates points, saves to history"""
    elapsed = time.time() - st.session_state.round_start_time
    tl = max(0, ROUND_TIME - int(elapsed))

    results = st.session_state.ai_results
    drawing = get_drawing(canvas_result) if canvas_result else None
    if drawing:
        results = infer_or_simulate(drawing, st.session_state.target_word)

    hit = any(r["top_guess"] == st.session_state.target_word for r in results)
    pts = max(10, tl * 10) if hit else 0

    st.session_state.score += pts
    st.session_state.round_history.append({
        "word": st.session_state.target_word,
        "correct": hit,
        "points": pts,
        "models": [
            {"model": r["model"], "guess": r["top_guess"],
             "conf": r["top_conf"], "correct": r["top_guess"] == st.session_state.target_word,
             "color": r["color"]}
            for r in results
        ] if results else [],
    })

    if st.session_state.round >= TOTAL_ROUNDS:
        st.session_state.game_state = "game_over"
    else:
        st.session_state.game_state = "round_end"


# menu screen
def screen_menu():
    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="label-sm" style="text-align:center; margin-bottom:8px;">CSC 580 — Sketch Recognition</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="title-main">Pictionary</h1>', unsafe_allow_html=True)
    st.markdown('<p class="title-sub">You draw. Three neural networks try to guess what it is.</p>', unsafe_allow_html=True)

    cols = st.columns([1, 3, 3, 3, 1])
    for i, (name, _, _, color) in enumerate(MODELS):
        with cols[i + 1]:
            st.markdown(f"""
            <div class="paper-card" style="text-align:center; padding:18px 12px;">
                <span class="model-tag" style="background:{color}18; color:{color} !important; font-size:13px;">{name}</span>
                <div style="font-size:12px; color:#6b6560; margin-top:8px;">{MODEL_DESCS[name]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    models = load_models()
    if models:
        st.markdown(f'<p style="text-align:center; font-size:12px; color:#2d6a4f !important;">Models loaded: {", ".join(models.keys())}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="text-align:center; font-size:12px; color:#b8860b !important;">No checkpoint files in models/ yet — using placeholder guesses for now</p>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _, bc, _ = st.columns([2, 1, 2])
    with bc:
        if st.button("Start Game", use_container_width=True, type="primary"):
            st.session_state.game_state = "playing"
            st.session_state.round = 1
            st.session_state.score = 0
            st.session_state.round_history = []
            st.session_state.used_words = []
            start_round()
            st.rerun()

    st.markdown(f'<p style="text-align:center; font-size:12px; color:#b5aea5; margin-top:12px;">{TOTAL_ROUNDS} rounds, {ROUND_TIME} seconds each, 50 categories</p>', unsafe_allow_html=True)


# the drawing screen where you actually play
def screen_playing():
    # if streamlit-autorefresh is installed, use it to tick the timer
    # otherwise show a manual refresh button
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=2000, key="playing_refresh")

    elapsed = time.time() - st.session_state.round_start_time
    tl = max(0, ROUND_TIME - int(elapsed))
    tc = "timer-ok" if tl > 15 else ("timer-warn" if tl > 7 else "timer-crit")

    # top bar
    c1, c2, c3 = st.columns([3, 5, 3])
    with c1:
        st.markdown(f"""
        <div style="padding-top:6px;">
            <span style="font-family:'Libre Baskerville',serif; font-weight:700; font-size:16px;">Pictionary</span>
            <span style="font-size:12px; color:#b5aea5; margin-left:10px;">Round {st.session_state.round} of {TOTAL_ROUNDS}</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f'<p class="label-sm" style="margin-bottom:2px;">Draw this</p><p class="draw-this">{st.session_state.target_word}</p>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div style="text-align:right; padding-top:6px;"><span class="score-pill">{st.session_state.score} pts</span></div>', unsafe_allow_html=True)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    col_draw, col_ai = st.columns([3, 2], gap="large")

    with col_draw:
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:6px;">
            <span class="timer-num {tc}">{tl}</span>
            <p class="label-sm" style="margin-top:2px;">seconds</p>
        </div>""", unsafe_allow_html=True)

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=st.session_state.brush_size,
            stroke_color="#2c2c2c",
            background_color="#fffdf8",
            width=CANVAS_SIZE, height=CANVAS_SIZE,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_id}",
        )

        cc1, cc2, cc3 = st.columns([3, 1, 1])
        with cc1:
            st.session_state.brush_size = st.slider("Brush", 2, 24, st.session_state.brush_size, label_visibility="collapsed")
        with cc2:
            if st.button("Clear", use_container_width=True):
                st.session_state.canvas_id += 1
                st.rerun()
        with cc3:
            if st.button("Submit", use_container_width=True, type="primary"):
                end_round(canvas_result)
                st.rerun()

        # manual refresh button if autorefresh isnt installed
        if not HAS_AUTOREFRESH:
            if st.button("Refresh guesses", use_container_width=True):
                st.rerun()

    with col_ai:
        st.markdown('<p class="label-sm">What the models see</p>', unsafe_allow_html=True)

        drawing = get_drawing(canvas_result)
        if drawing:
            # this is the 28x28 image that actually gets fed into the networks
            st.image(drawing.resize((28, 28)), caption="28 × 28 input", width=84)
            results = infer_or_simulate(drawing, st.session_state.target_word)
            st.session_state.ai_results = results
            for r in results:
                model_card(r, st.session_state.target_word)
        else:
            st.markdown('<div class="waiting-msg">Start sketching and the<br>models will guess in real time.</div>', unsafe_allow_html=True)

        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        for name, _, _, color in MODELS:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:3px;">
                <div style="width:8px; height:8px; border-radius:2px; background:{color};"></div>
                <span style="font-size:11px; color:#6b6560;">{name} — {MODEL_DESCS[name]}</span>
            </div>""", unsafe_allow_html=True)

    # auto submit when time runs out
    if tl <= 0:
        end_round(canvas_result)
        st.rerun()


# shows results after each round
def screen_round_end():
    last = st.session_state.round_history[-1]

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    st.markdown(f'<p class="label-sm" style="text-align:center;">Round {st.session_state.round} of {TOTAL_ROUNDS}</p>', unsafe_allow_html=True)

    verdict = "At least one model got it right." if last["correct"] else "None of the models guessed correctly."
    vcolor = "#2d6a4f" if last["correct"] else "#c0583a"
    st.markdown(f"""
    <div style="text-align:center; margin:12px 0 28px;">
        <p style="font-size:16px; color:#6b6560;">The word was</p>
        <p style="font-family:'Libre Baskerville',serif; font-size:44px; font-weight:700;
                  font-style:italic; text-transform:capitalize; margin:4px 0 12px;">{last['word']}</p>
        <p style="font-size:18px; color:{vcolor};">{verdict}</p>
    </div>""", unsafe_allow_html=True)

    if last["models"]:
        cols = st.columns([1, 3, 3, 3, 1])
        for i, m in enumerate(last["models"]):
            with cols[i + 1]:
                border = "#2d6a4f" if m["correct"] else "#ddd5c8"
                gcol = "#2d6a4f" if m["correct"] else "#6b6560"
                st.markdown(f"""
                <div class="paper-card" style="text-align:center; border-color:{border};">
                    <span class="model-tag" style="background:{m['color']}18; color:{m['color']} !important;">{m['model']}</span>
                    <div style="font-size:14px; margin-top:8px; text-transform:capitalize; color:{gcol};">{m['guess']}</div>
                    <div style="font-size:12px; color:#b5aea5; margin-top:2px;">{round(m['conf'] * 100)}% confidence</div>
                </div>""", unsafe_allow_html=True)

    pts_color = "#2d6a4f" if last["points"] > 0 else "#b5aea5"
    st.markdown(f"""
    <div style="text-align:center; margin-top:20px;">
        <span class="score-pill">{st.session_state.score} pts total</span>
        <span style="margin-left:12px; font-size:14px; color:{pts_color}; font-weight:500;">+{last['points']} this round</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    _, bc, _ = st.columns([2, 1, 2])
    with bc:
        if st.button("Next Round", use_container_width=True, type="primary"):
            st.session_state.round += 1
            start_round()
            st.session_state.game_state = "playing"
            st.rerun()


# final score screen
def screen_game_over():
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="label-sm" style="text-align:center;">Game Over</p>', unsafe_allow_html=True)

    correct_count = sum(1 for r in st.session_state.round_history if r["correct"])
    st.markdown(f"""
    <div style="text-align:center; margin:12px 0 32px;">
        <p style="font-family:'Libre Baskerville',serif; font-size:52px; font-weight:700; margin:0;">{st.session_state.score}</p>
        <p style="font-size:14px; color:#6b6560; margin-top:4px;">points — {correct_count} of {TOTAL_ROUNDS} rounds guessed correctly</p>
    </div>""", unsafe_allow_html=True)

    for r in st.session_state.round_history:
        cls = "result-row result-row-hit" if r["correct"] else "result-row result-row-miss"
        bg = "#2d6a4f" if r["correct"] else "#c0583a"
        sym = "✓" if r["correct"] else "✗"
        mtext = ", ".join(f'{m["model"]}: {m["guess"]}' for m in r["models"]) if r["models"] else ""
        pcol = "#2d6a4f" if r["points"] > 0 else "#b5aea5"

        st.markdown(f"""
        <div class="{cls}">
            <div class="result-icon" style="background:{bg};">{sym}</div>
            <div style="flex:1;">
                <div style="font-weight:700; font-size:14px; text-transform:capitalize;">{r['word']}</div>
                <div style="font-size:11px; color:#b5aea5;">{mtext}</div>
            </div>
            <div style="font-weight:700; font-size:13px; color:{pcol};">+{r['points']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    _, bc, _ = st.columns([2, 1, 2])
    with bc:
        if st.button("Play Again", use_container_width=True, type="primary"):
            st.session_state.game_state = "menu"
            st.session_state.round = 0
            st.session_state.score = 0
            st.session_state.round_history = []
            st.session_state.used_words = []
            st.session_state.ai_results = []
            st.rerun()


# route to whatever screen we're on
screen = {
    "menu": screen_menu,
    "playing": screen_playing,
    "round_end": screen_round_end,
    "game_over": screen_game_over,
}
screen[st.session_state.game_state]()