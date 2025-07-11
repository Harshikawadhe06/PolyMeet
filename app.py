import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import tempfile
import os

# Custom CSS for dark theme and background
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: #f5f5f5;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1581092334399-fd1cc0faa0f6");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(30, 30, 30, 0.8);
        padding: 2rem;
        border-radius: 10px;
    }
    .stTabs [role="tab"] {
        background-color: #2d2d2d;
        color: white;
        border-radius: 10px 10px 0 0;
        padding: 10px;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #444444;
        font-weight: bold;
        border-bottom: 3px solid #00ffcc;
    }
    </style>
""", unsafe_allow_html=True)

# Set page config
st.set_page_config(page_title="PolyMeet Pro", layout="centered")

st.title("🎧 PolyMeet Pro – Smart Meeting Summarizer")

st.markdown("""
Upload your `.wav` meeting audio to get a detailed **Transcript**, **Summary**, and **To-Do List** – all styled for productivity.
""")

# Load summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# File uploader
uploaded_file = st.file_uploader("Upload a WAV file (Max: 1 minute)", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(temp_audio_path) as source:
            with st.spinner("🔍 Transcribing..."):
                audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data)

        with st.spinner("🧠 Generating summary..."):
            summary = summarizer(transcript, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
            todo_prompt = f"From this meeting text, extract a to-do list:\n{transcript}\nTasks:"
            tasks = summarizer(todo_prompt, max_length=100, min_length=40, do_sample=False)[0]['summary_text']

        # Tabs for organized layout
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Transcript", "🧠 Summary", "✅ To-Do List", "📅 Schedule"])

        with tab1:
            st.subheader("📝 Full Transcript")
            st.text_area("Transcript", transcript, height=300)
            st.download_button("⬇ Download Transcript", transcript, file_name="transcript.txt")

        with tab2:
            st.subheader("🧠 Key Points Summary")
            st.markdown(f"> {summary}")
            st.download_button("⬇ Download Summary", summary, file_name="summary.txt")

        with tab3:
            st.subheader("✅ Actionable Tasks")
            st.markdown("- " + tasks.replace(". ", ".\n- "))
            st.download_button("⬇ Download To-Do List", tasks, file_name="todo_list.txt")

        with tab4:
            st.subheader("📅 Schedule (Coming Soon)")
            st.info("⏳ Auto schedule extraction from transcript will be added in the next version.")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

    # Clean up temp file
    try:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
    except Exception as delete_error:
        st.warning(f"⚠️ Couldn't delete temp file: {delete_error}")
