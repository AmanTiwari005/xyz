import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from gtts import gTTS
import os
import requests
import openai
from tempfile import NamedTemporaryFile
import en_core_web_sm

# Load spaCy model
nlp = en_core_web_sm.load()

# Initialize summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# OpenAI API Key from Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to get YouTube transcript
def get_transcript(video_url):
    try:
        video_id = video_url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript, video_id
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None, None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None, None

# Function to extract key highlights & insights
def extract_keywords(text):
    doc = nlp(text)
    keywords = {ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "GPE", "PERSON", "EVENT"]}
    return list(keywords)

# Function to generate structured summaries
def generate_summary(transcript):
    try:
        text = " ".join([t["text"] for t in transcript])
        max_chunk_length = 1024
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        summaries = [summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
        full_summary = " ".join(summaries)
        keywords = extract_keywords(full_summary)
        highlights = [f"✅ {kw}" for kw in keywords]
        return full_summary, highlights
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return None, None

# Function to convert summary to speech
def text_to_speech(summary):
    tts = gTTS(text=summary, lang='en')
    temp_audio = NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# Chatbot function using OpenAI
def chat_with_ai(transcript, user_question):
    try:
        transcript_text = " ".join([t["text"] for t in transcript])[:4000]
        prompt = f"Based on this video transcript, answer the following question concisely:\n\nTranscript:\n{transcript_text}\n\nQuestion: {user_question}\nAnswer:"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on a video transcript."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# Streamlit app
def main():
    st.title("🎥 AI YouTube Video Summarizer & Chatbot")
    st.markdown("Summarize YouTube videos into **highlights, key insights, and audio** and chat with an AI about the video!")

    video_url = st.text_input("🎥 Enter YouTube Video URL:")
    if st.button("Summarize"):
        if video_url:
            with st.spinner("Fetching transcript and summarizing..."):
                transcript, _ = get_transcript(video_url)
                if transcript:
                    summary, highlights = generate_summary(transcript)
                    if summary:
                        st.subheader("📄 Summary")
                        st.write(summary)
                        st.subheader("📌 Highlights")
                        st.markdown("<br>".join(highlights), unsafe_allow_html=True)
                        audio_file = text_to_speech(summary)
                        st.audio(audio_file, format="audio/mp3")
        else:
            st.warning("⚠️ Please enter a valid YouTube video URL.")

    if 'transcript' in locals() and transcript:
        st.subheader("💬 Ask the AI About This Video")
        user_question = st.text_input("Ask a question about this video:")
        if user_question:
            response = chat_with_ai(transcript, user_question)
            st.write(f"**🤖 AI Answer:** {response}")

if __name__ == "__main__":
    main()
