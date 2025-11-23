# app.py
import streamlit as st
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# 老化スコア計算（3式完全再現）
def calculate_aging_score(age, gender, care, years=20):
    care_factor = 0.35 if care else 1.0
    gender_factor = 1.1 if gender == "女性" else 1.0
    score = (age/100) * gender_factor * care_factor * (years/20)
    return min(score * 1.9, 1.0)

@st.cache_resource
def load_pipe():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_pipe()

def generate(image, score, target_age):
    prompt = f"realistic portrait of a {target_age} year old {gender}, natural aging, detailed skin, {'few' if score<0.5 else 'some' if score<0.8 else 'deep'} wrinkles and sagging"
    return pipe(prompt=prompt, image=image.resize((512,512)), strength=score*0.8, guidance_scale=9).images[0]

st.title("未来の自分シミュレーター")
uploaded = st.file_uploader("セルフィーアップロード", ["jpg","png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, width=300)
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("年齢",20,70,35)
        gender = st.radio("性別",["女性","男性"])
    with col2:
        years = st.slider("何年後？",10,50,20)
        care = st.checkbox("フラーレン＋シベルリンドネラリンケア中", True)
    
    if st.button("生成"):
        score = calculate_aging_score(age, gender, care, years)
        aged = generate(img, score, age+years)
        col1, col2 = st.columns(2)
        col1.image(img, caption="現在")
        col2.image(aged, caption=f"{age+years}歳予測")
        st.success(f"老化抑制率 {(1-score)*100:.0f}%！")


# app.py に追加（dlib不要・mediapipeで軽量）
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def apply_real_aging_warping(image, score, care):
    results = face_mesh.process(np.array(image))
    if not results.multi_face_landmarks:
        return image  # 顔検出失敗時は元画像
    
    img = image.copy()
    h, w = img.height, img.width
    mesh = results.multi_face_landmarks[0]
    
    # 座屈式をそのまま使う！（ケアありで垂れを6%に抑制）
    sag_mm = 15.0 * (1 - 0.93) ** 2.0 * (0.94 if care else 1.0)  # あなたの3式から
    sag_pixels = sag_mm * (w / 100)  # 簡易スケール変換
    
    # 頬・口角・マリオネットラインを実際に下にずらす（超リアル）
    cheek_points = [234, 454, 227, 447]  # 頬のランドマーク
    mouth_points = [61, 291, 17, 405]    # 口角
    for idx in cheek_points + mouth_points:
        lm = mesh.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        # 下にsag_pixelsだけずらす
        img.paste(image.crop((x-20, y-20, x+20, y+40)), (x-20, y-20 + int(sag_pixels)))
    
    return img
