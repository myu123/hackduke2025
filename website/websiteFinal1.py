import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time
from PIL import Image
import torch
import base64
import os

def get_file_path(filename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

def get_base64(file_name: str) -> str:
    path_to_file = get_file_path(file_name)
    with open(path_to_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def transform(signal):
    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals)
    half = len(fft_vals) // 2
    freq_axis = np.fft.fftfreq(len(signal), d=1)[:half]
    return freq_axis, fft_vals[:half]

@st.cache_resource
def load_model():
    from model_architecture import HybridZulfModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridZulfModel(input_channels=2, seq_length=1024).to(device)
    checkpoint = torch.load(get_file_path("best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device

# Build absolute path for images
background_base64 = get_base64("right.png")
site_logo_path = get_file_path("site_logo.png")

st.set_page_config(
    page_title="Zulf-NMR Spectrum Predictor",
    page_icon=Image.open(site_logo_path),
)

global_css = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("data:image/png;base64,{background_base64}") no-repeat center center fixed;
    background-size: cover;
}}
body {{
    font-family: 'Roboto', sans-serif;
    color: #333333;
}}
h1, h2, h3, h4 {{
    color: #034f84;
}}
a {{
    color: #034f84;
    text-decoration: none;
}}
a:hover {{
    text-decoration: underline;
}}
div.stButton > button {{
    background-color: #034f84;
    color: #ffffff;
    border: none;
    border-radius: 5px;
    padding: 0.5em 1em;
    font-size: 1em;
    transition: background-color 0.3s ease;
}}
div.stButton > button:hover {{
    background-color: #026c9e;
}}
.css-1hynsf3.e1fqkh3o1 {{
    background-color: #e8f4f8;
}}
.professor-banner {{
    background-color: #e8f4f8;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}}
.professor-banner h2 {{
    color: #034f84;
    font-weight: bold;
    font-size: 2em;
}}
.professor-banner p {{
    color: #034f84;
    font-size: 1.2em;
    font-weight: bold;
    margin-top: -10px;
}}
.professor-banner a {{
    font-size: 1em;
    color: #034f84;
    text-decoration: underline;
}}
@media (prefers-color-scheme: dark) {{
    .professor-banner {{
        background-color: #3a3a3a;
    }}
    .professor-banner h2,
    .professor-banner p {{
        color: #ffffff !important;
    }}
    .professor-banner a {{
        color: #90caf9 !important;
    }}
}}
</style>
"""
st.markdown(global_css, unsafe_allow_html=True)

def main():
    with st.sidebar:
        st.markdown("""
        <div style="background-color: #034f84; padding: 10px; border-radius: 10px; text-align: center;">
            <h3 style="color: #fff;">Welcome!</h3>
            <p style="color: #fff;">NMR Spectroscopy: Nuclear Resonance Sequence Conversion</p>
        </div>
        """, unsafe_allow_html=True)
        logo_path = get_file_path("site_logo.png")
        st.image(logo_path, use_container_width=True)

        st.markdown("### Model Info")
        st.write("**Hybrid Zulf Model 1.0**")
        st.write("Trained to convert ultra-low field NMR signals into high-field spectra.")
        st.markdown("### How It Works")
        st.write("Upload a fid file with at least three columns (Time, X, Y) **or** use sample data. The model processes the fid file and generates a predicted high-field spectrum.")

        # If you have a QR code image, do the same approach:
        qr_path = get_file_path("site_qr_code.png")
        try:
            st.image(qr_path, use_container_width=True)
        except FileNotFoundError:
            st.warning("QR code image not found. Skipping display.")

        st.markdown("<p style='text-align: center; font-style: italic;'>Fun Fact: Out of the 9162 lines of code we wrote, only 855 made the cut!</p>", unsafe_allow_html=True)

    tabs = st.tabs(["Home", "About", "Meet the Team"])
    with tabs[0]:
        home()
    with tabs[1]:
        about()
    with tabs[2]:
        bios()

def home():
    seq_length = 1024
    professor_banner = """
    <div class="professor-banner">
        <h2>“imagine knowing what's in your food using just your phone”</h2>
        <p>Thomas Theis</p>
        <a href="https://chemistry.sciences.ncsu.edu/people/ttheis/" target="_blank">
           Professor Thomas Theis, Department of Chemistry at NC State University
        </a>
    </div>
    """
    st.markdown(professor_banner, unsafe_allow_html=True)

    st.title("Conversion Demo")
    st.write("**Instructions:**  \nUpload your Zulf-NMR fid file below **or** use the sample data option if you don't have a file.")

    use_sample = st.checkbox("Use sample data (random_zulf_fid_sample.txt) instead of uploading a file")

    uploaded_file = None
    if not use_sample:
        uploaded_file = st.file_uploader("Upload your Zulf-NMR fid file", type=["fid", "txt", "dat", "npy"])
        if uploaded_file is not None:
            st.write("File uploaded:", uploaded_file.name)
    else:
        st.write("Sample data will be used: **random_zulf_fid_sample.txt**")

    if "prev_mode" not in st.session_state:
        st.session_state["prev_mode"] = None
    if "prev_file" not in st.session_state:
        st.session_state["prev_file"] = None

    current_mode = "sample" if use_sample else "upload"
    current_file = uploaded_file.name if uploaded_file else None

    if current_mode != st.session_state["prev_mode"] or current_file != st.session_state["prev_file"]:
        st.session_state.pop("predicted_df", None)
        st.session_state.pop("fft_df", None)
        st.session_state["prev_mode"] = current_mode
        st.session_state["prev_file"] = current_file

    if st.button("Generate High-field Spectrum"):
        with st.spinner("Processing file and generating high-field spectrum..."):
            import torch

            if use_sample:
                try:
                    sample_path = get_file_path("random_zulf_fid_sample.txt")
                    data = np.loadtxt(sample_path, skiprows=1)
                except Exception as e:
                    st.error("Could not load sample file: " + str(e))
                    return
            else:
                if uploaded_file is None:
                    st.error("No file uploaded. Please upload a file or select 'Use sample data'.")
                    return
                try:
                    if uploaded_file.name.endswith(".npy"):
                        data = np.load(uploaded_file)
                    else:
                        try:
                            data = np.loadtxt(uploaded_file)
                        except ValueError:
                            uploaded_file.seek(0)
                            data = np.loadtxt(uploaded_file, skiprows=1)
                except Exception as e:
                    st.error("Error reading the file: " + str(e))
                    return

            if len(data.shape) == 1 or data.shape[1] < 3:
                st.error("Data format error: expected (N, >=3) shape [Time, Real, Imag].")
                return

            fid_signal = data[:, 1:3]
            num_points = fid_signal.shape[0]
            if num_points >= seq_length:
                fid_input = fid_signal[:seq_length]
            else:
                pad_length = seq_length - num_points
                fid_input = np.pad(fid_signal, ((0, pad_length), (0, 0)), mode='constant')

            fid_input = fid_input[np.newaxis, :, :]
            fid_input = np.transpose(fid_input, (0, 2, 1))

            model, device = load_model()
            input_tensor = torch.tensor(fid_input, dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = model(input_tensor)
                difference = pred.squeeze(-1)[0]

            baseline_tensor = input_tensor[0, 0, :]
            baseline = baseline_tensor.cpu().numpy()
            difference_np = difference.cpu().numpy()
            reconstructed_fid = baseline + difference_np

            df_final = pd.DataFrame({
                "Sample Index": np.arange(seq_length),
                "High-field FID": reconstructed_fid
            })

            anim_container = st.empty()
            num_frames = 20
            for i in range(1, num_frames + 1):
                fraction = i / num_frames
                df_temp = df_final.copy()
                df_temp["High-field FID"] *= fraction
                chart_temp = alt.Chart(df_temp).mark_line(color='red', strokeWidth=2).encode(
                    x=alt.X("Sample Index", title="Sample Index"),
                    y=alt.Y("High-field FID", title="Intensity (a.u.)"),
                    tooltip=["Sample Index", "High-field FID"]
                ).properties(width=700, height=300)
                anim_container.altair_chart(chart_temp, use_container_width=True)
                time.sleep(0.1)
            anim_container.empty()
            st.success("High-field NMR spectrum generated (time-domain)!")

            freq_axis, spec_vals = transform(reconstructed_fid)
            df_fft = pd.DataFrame({
                "Frequency": freq_axis,
                "Amplitude": spec_vals
            })

            st.session_state["predicted_df"] = df_final
            st.session_state["fft_df"] = df_fft

    if "predicted_df" in st.session_state and "fft_df" in st.session_state:
        df_final = st.session_state["predicted_df"]
        df_fft = st.session_state["fft_df"]

        st.markdown("### Frequency-Domain (Fourier Transform)")
        fft_chart = alt.Chart(df_fft).mark_line(color='red', strokeWidth=2).encode(
            x=alt.X("Frequency", title="Frequency (arbitrary units)"),
            y=alt.Y("Amplitude", title="Magnitude"),
            tooltip=["Frequency", "Amplitude"]
        ).properties(width=700, height=300).interactive()
        st.altair_chart(fft_chart, use_container_width=True)

        st.markdown("### Final Time-Domain Result (FID)")
        base_line = alt.Chart(df_final).mark_line(color='red', strokeWidth=2).encode(
            x="Sample Index",
            y="High-field FID",
            tooltip=["Sample Index", "High-field FID"]
        ).properties(width=700, height=300).interactive()
        st.altair_chart(base_line, use_container_width=True)

        st.write("### Data Preview (Time-Domain Prediction)")
        start_idx, end_idx = st.slider(
            "Select sample index range",
            0, seq_length - 1, (0, min(100, seq_length-1)), key="data_slider"
        )
        st.dataframe(df_final.iloc[start_idx:end_idx + 1])

def about():
    st.title("About")
    st.markdown("""
## Inspiration
My brother is an odd guy...
""", unsafe_allow_html=False)

def bios():
    st.title("🖥️ Team Bios")
    st.markdown("""
    <style>
        .streamlit-expanderHeader {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
        }
        .bio-card:hover {
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.expander("🎸 Max - Python Enjoyer", expanded=True):
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                try:
                    st.image(get_file_path("hackdukeMax.png"), width=150)
                except FileNotFoundError:
                    st.warning("Max avatar image not found!")
            with col2:
                st.markdown("""
                **🎓 University:** UNC Chapel Hill  
                **💻 Favorite Languages:** Python, Swift  
                **🏆 Hobbies:**  
                - Guitar  
                - Coding  
                - Gaming  
                **🧏‍♂️ Worked On:** Complete website development and integration of model; optimizatin of model
                """)

    with st.expander("♜ Seth - Data Enthusiast", expanded=True):
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                try:
                    st.image(get_file_path("hackdukeSeth.png"), width=150)
                except FileNotFoundError:
                    st.warning("Seth avatar image not found!")
            with col2:
                st.markdown("""
                **🎓 University:** UNC Chapel Hill  
                **💻 Favorite Languages:** Python, C    
                **🏆 Hobbies:**  
                - Chess  
                - Coding  
                - Basketball  
                **🤕 Worked On:** Developed deep learning model; designed and executed data generation
                """)

    st.divider()
    with st.container():
        st.markdown("<h2 style='text-align: center;'>🚨 Honorary Third Team Member 🚨</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 3])
        with col1:
            try:
                st.image(get_file_path("isaac_thumbs_up.png"), caption="Professional basement explorer since 2011", use_container_width=True)
            except FileNotFoundError:
                st.error("🚫 Missing crucial team member! (Did the basement door close?)")
        with col2:
            st.markdown("""
            ### 🎮 Isaac "The Binding" Smith
            **Official Title:** Eternal Runs Champion  
            **Alignment:** Chaotic Neutral  
            **Current Location:** Depths II  
            **Special Skills:**
            - Finding breakfast in item rooms 🥞
            - Dodging mom's foot 👟
            - Converting tears into DPS 💧➔💥
            """)
            st.progress(0.9, text="Tear Production Capacity")
            st.progress(1.0, text="Salt Generation (Team Meetings)")
            st.progress(0.4, text="Touch Grass Initiative")
            st.caption("⚠️ Warning: May randomly spawn with Dr. Fetus or Soy Milk")

    st.markdown("""
    <div style='text-align: center; margin-top: 20px; font-size: 0.8em; color: #666;'>
    *Isaac's participation pending successful escape from basement.  
    Team not responsible for swallowed pennies or accidental polytheism.*
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
