import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time

def main():
    tabs = st.tabs(["Home", "About"])
    
    with tabs[0]:
        home()
        
    with tabs[1]:
        about()

def home():
    st.title("NMR / Zulf-NMR Spectroscopy Prediction Demo")
    st.write("Enter the J-couplings (in Hz) below and click **Predict Spectrum** to see the predicted spectrum.")
    
    j_input = st.text_input("Enter J-couplings (separated by commas)", "7.0, 2.5, 9.8")
    
    if st.button("Predict Spectrum"):
        try:
            couplings = [float(val.strip()) for val in j_input.split(",") if val.strip() != ""]
            st.write("J-couplings:", couplings)
        except ValueError:
            st.error("Please enter valid numeric values separated by commas.")
            return
        
        st.write("Generating predicted spectrum...")
        
        placeholder = st.empty()
        
        x = np.linspace(0, 10, 500)
        
        y_final = np.zeros_like(x)
        num_peaks = np.random.randint(2, 5)  
        for _ in range(num_peaks):
            amplitude = np.random.uniform(1, 5)
            center = np.random.uniform(0, 10)
            width = np.random.uniform(0.1, 1.0)
            y_final += amplitude * np.exp(-((x - center)**2) / (2 * width**2))
        

        y_final = y_final / np.max(y_final)
        
    
        num_frames = 20
        for i in range(1, num_frames + 1):
            fraction = i / num_frames
            y_current = y_final * fraction
            df = pd.DataFrame({
                "Chemical Shift (ppm)": x,
                "Intensity": y_current
            })
            
            
            # it's common to display NMR spectra with the chemical shift axis reversed
            chart = alt.Chart(df).mark_line(color='steelblue', strokeWidth=2).encode(
                x=alt.X("Chemical Shift (ppm)", scale=alt.Scale(domain=[10, 0])),
                y=alt.Y("Intensity", title="Intensity (a.u.)")
            ).properties(
                width=700,
                height=400
            )
            
            placeholder.altair_chart(chart, use_container_width=True)
            time.sleep(0.1)
        
        st.success("Spectrum prediction complete!")

def about():
    st.title("About")
    st.write("""
        I love NMR/Zulf-NMR!!!!!
    """)

if __name__ == "__main__":
    main()