import streamlit as st
import os
from model import detect

def save_uploaded_file(uploaded_file, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    with open(os.path.join(save_directory, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Video saved successfully to {save_directory}/{uploaded_file.name}")
    

def main():
    status = False
    st.title("Obstacle detection in Indoor Spaces")
    uploaded_file = st.file_uploader("Input Indoor Video file...", type=["mp4"])
    save_directory = "archive"
    
    if uploaded_file is not None:
        
        row = st.columns(2)
        with row[0].container():
            video_bytes = uploaded_file.read()
            with st.container(height=600):
                st.video(video_bytes)
                print(uploaded_file.name)

            if st.button("Run Object Detection"):
                if uploaded_file is not None:
                    save_uploaded_file(uploaded_file, save_directory)
                else:
                    st.error("Please upload a video file first.")

                status = detect(uploaded_file)
        with row[1].container():       
            if status:
                with st.container(height=600):
                    video_file = open('C:/Users/d4rsh/Downloads/agile od dataset/archive/live_test/sample_pred.mp4', 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                pass
                    
            


if __name__ == "__main__":
    main()
