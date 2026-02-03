# TO RUN: streamlit run app.py

import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="PictionAIry", layout="centered")
st.title("PictionAIry")

if "canvas_id" not in st.session_state:
    st.session_state.canvas_id = 0

st.sidebar.header("Controls")
brush_size = st.sidebar.slider("Brush size", 2, 40, 12)

if st.sidebar.button("Clear"):
    st.session_state.canvas_id += 1 

canvas_size = 400
bg_color = "#FFFFFF"
stroke_color = "#000000"

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=brush_size,
    stroke_color=stroke_color,
    background_color=bg_color,
    width=canvas_size,
    height=canvas_size,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_id}", 
)

st.subheader("What the AI sees")
if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)  # RGBA
    pil_img = Image.fromarray(img, mode="RGBA").convert("L")
    small = pil_img.resize((28, 28))
    arr = np.array(small)

    st.image(pil_img, caption="Grayscale capture", width=200)
    st.write(arr[:8, :8])
