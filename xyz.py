import streamlit as st 
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from gtts import gTTS
import os
import requests
import spacy
import openai  # OpenAI Chat API for chatbot

# Load API keys from environment variables
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Load NLP model for keyword extraction
nlp = spacy.load("en_core_web_sm")

# Initialize summarization and chatbot models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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

# Function to fetch video metadata
def get_video_info(video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={YOUTUBE_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        video_info = response.json()
        if 'items' in video_info and len(video_info['items']) > 0:
            title = video_info['items'][0]['snippet']['title']
            channel = video_info['items'][0]['snippet']['channelTitle']
            description = video_info['items'][0]['snippet']['description']
            return title, channel, description
    return None, None, None

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
        insights_prompt = f"Extract 2-3 insightful points from this summary: {full_summary}"
        insights = summarizer(insights_prompt, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        
        return full_summary, highlights, insights
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return None, None, None

# Function to convert summary to speech
def text_to_speech(summary):
    tts = gTTS(text=summary, lang='en')
    audio_path = "/tmp/summary.mp3"
    tts.save(audio_path)
    return audio_path

# Chatbot function using OpenAI
def chat_with_ai(transcript, user_question):
    try:
        transcript_text = " ".join([t["text"] for t in transcript])[:3000]  # Reduced token limit
        prompt = f"Based on this video transcript, answer the following question concisely:\n\nTranscript:\n{transcript_text}\n\nQuestion: {user_question}\nAnswer:"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant that answers questions based on a video transcript."},
                      {"role": "user", "content": prompt}]
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
                transcript, video_id = get_transcript(video_url)
                if transcript:
                    title, channel, description = get_video_info(video_id)
                    if title and channel:
                        st.subheader("🎬 Video Information")
                        st.write(f"**📌 Title:** {title}")
                        st.write(f"**📺 Channel:** {channel}")
                        st.write(f"**📄 Description:** {description[:300]}...")
                    
                    summary, highlights, key_insights = generate_summary(transcript)
                    if summary:
                        st.success("✅ Summary generated successfully!")
                        st.subheader("📄 Summary")
                        st.write(summary)
                        st.subheader("📌 Highlights")
                        st.markdown("<br>".join(highlights), unsafe_allow_html=True)
                        st.subheader("💡 Key Insights")
                        st.markdown(f"💡 {key_insights.replace('. ', '<br>💡 ')}", unsafe_allow_html=True)
                        
                        audio_file = text_to_speech(summary)
                        st.audio(audio_file, format="audio/mp3")
                        
                        st.subheader("💬 Ask the AI About This Video")
                        user_question = st.text_input("Ask a question about this video:")
                        if user_question:
                            response = chat_with_ai(transcript, user_question)
                            st.write(f"**🤖 AI Answer:** {response}")
        else:
            st.warning("⚠️ Please enter a valid YouTube video URL.")

if __name__ == "__main__":
    main()
