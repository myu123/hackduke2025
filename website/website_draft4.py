import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time
from PIL import Image

st.set_page_config(
    page_title="Zulf-NMR Spectrum Predictor",
    page_icon="🧪",
)

global_css = """
<style>
/* Global body settings */
body {
    font-family: 'Roboto', sans-serif;
    background-color: #f8f9fa;
    color: #333333;
}

/* Headings */
h1, h2, h3, h4 {
    color: #034f84;
}

/* Link styling */
a {
    color: #034f84;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}

/* Button styling */
div.stButton > button {
    background-color: #034f84;
    color: #ffffff;
    border: none;
    border-radius: 5px;
    padding: 0.5em 1em;
    font-size: 1em;
    transition: background-color 0.3s ease;
}
div.stButton > button:hover {
    background-color: #026c9e;
}

/* Tab header background */
.css-1hynsf3.e1fqkh3o1 {
    background-color: #e8f4f8;
}

/* Professor banner styling */
.professor-banner {
    background-color: #e8f4f8;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}
.professor-banner h2 {
    color: #034f84;
    font-weight: bold;
    font-size: 2em;
}
.professor-banner p {
    color: #034f84;
    font-size: 1.2em;
    font-weight: bold;
    margin-top: -10px;
}
.professor-banner a {
    font-size: 1em;
    color: #034f84;
    text-decoration: underline;
}

/* Dark theme adjustments for professor banner */
@media (prefers-color-scheme: dark) {
    .professor-banner {
        background-color: #3a3a3a;
    }
    .professor-banner h2,
    .professor-banner p {
        color: #ffffff !important;
    }
    .professor-banner a {
        color: #90caf9 !important;
    }
}

/* Additional customizations can be added below */
</style>
"""
st.markdown(global_css, unsafe_allow_html=True)

def main():
    tabs = st.tabs(["Home", "About", "Meet the Team"])
    
    with tabs[0]:
        home()
        
    with tabs[1]:
        about()

    with tabs[2]:
        bios()

def home():
    professor_banner = """
    <div class="professor-banner">
        <h2>“Imagine knowing what's in your food using your phone”</h2>
        <p>Thomas Theis</p>
        <a href="https://chemistry.sciences.ncsu.edu/people/ttheis/" target="_blank">
           Professor Thomas Theis, Department of Chemistry at NC State University
        </a>
    </div>
    """
    st.markdown(professor_banner, unsafe_allow_html=True)
    
    st.title("NMR / Zulf-NMR Spectroscopy Prediction Demo")
    st.write("""
        Upload your Zulf-NMR fid file below. 
        The file will be fed into a (simulated) model to generate a high-field/high-quality NMR spectrum.
    """)
    
    uploaded_file = st.file_uploader("Upload your Zulf-NMR fid file", type=["fid", "txt", "dat"])
    
    if uploaded_file is not None:
        st.write("File uploaded:", uploaded_file.name)
        
        if st.button("Generate High-field Spectrum"):
            with st.spinner("Processing file and generating spectrum..."):
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)  # simulate processing delay
                    progress_bar.progress(percent_complete + 1)
                
                x = np.linspace(0, 10, 500)
                y_final = np.zeros_like(x)
                num_peaks = np.random.randint(3, 6)
                for _ in range(num_peaks):
                    amplitude = np.random.uniform(1, 5)
                    center = np.random.uniform(0.5, 9.5)
                    width = np.random.uniform(0.05, 0.2)
                    y_final += amplitude * np.exp(-((x - center)**2) / (2 * width**2))
                
                y_final = y_final / np.max(y_final)
                
                placeholder = st.empty()
                num_frames = 20
                for i in range(1, num_frames + 1):
                    fraction = i / num_frames
                    y_current = y_final * fraction
                    df = pd.DataFrame({
                        "Chemical Shift (ppm)": x,
                        "Intensity": y_current
                    })
                    
                    # in NMR the chemical shift axis is often reversed
                    chart = alt.Chart(df).mark_line(color='darkgreen', strokeWidth=2).encode(
                        x=alt.X("Chemical Shift (ppm)", scale=alt.Scale(domain=[10, 0])),
                        y=alt.Y("Intensity", title="Intensity (a.u.)"),
                        tooltip=["Chemical Shift (ppm)", "Intensity"]
                    ).interactive().properties(
                        width=700,
                        height=400
                    )
                    
                    placeholder.altair_chart(chart, use_container_width=True)
                    time.sleep(0.1)
                
                st.success("High-field NMR spectrum generated!")
    else:
        st.info("Awaiting fid file upload.")

def about():
    st.title("About")
    st.write("""
        This demo app simulates the process of converting Zulf-NMR fid files into high-field/high-quality NMR spectra.
        In a production environment, the uploaded fid file would be processed by a machine learning model to generate an accurate prediction.
        For now, the app uses randomness to simulate the spectral output.
    """)

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

    # Max's Bio
    with st.expander("🎸 Max - Python Enjoyer", expanded=True):
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                try:
                    st.image("hackdukeMax.png", width=150)
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
                """)

    # Seth's Bio
    with st.expander("🚀 Seth - Data Enthusiast", expanded=True):
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                try:
                    st.image("hackdukeSeth.png", width=150)
                except FileNotFoundError:
                    st.warning("Seth avatar image not found!")
            with col2:
                st.markdown("""
                **🎓 University:** UNC Chapel Hill  
                **💻 Favorite Languages:** Python  
                **🏆 Hobbies:**  
                - Chess  
                - Coding  
                - Basketball  
                """)
    
    # complete fake member Isaac
    st.divider()
    with st.container():
        st.markdown("<h2 style='text-align: center;'>🚨 Honorary Third Team Member 🚨</h2>", 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            try:
                st.image("isaac_thumbs_up.png", 
                        caption="Professional basement explorer since 2011",
                        use_container_width=True)
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