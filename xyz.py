import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
from gtts import gTTS
import os
import tempfile
from moviepy.editor import TextClip, concatenate_videoclips, AudioFileClip

# Load the AI Model
@st.cache_resource
def load_model():
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

def generate_script(prompt, max_length=600):
    """Generates a longer movie script snippet based on the given prompt."""
    if not prompt.strip():
        return "Please provide a valid prompt."
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    script = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return script

# Extract skills/roles from the generated script
def extract_skills_from_script(script):
    """Extract possible skills/roles from the movie script."""
    roles_list = ["Acting", "Directing", "Editing", "Animation", "VFX", "Voiceover"]
    found_roles = []
    for role in roles_list:
        if re.search(r'\b' + re.escape(role) + r'\b', script, re.IGNORECASE):
            found_roles.append(role)
    return found_roles

# Function to convert text to audio using gTTS
def text_to_audio(text):
    """Converts the provided text to an audio file using gTTS."""
    tts = gTTS(text, lang='en')
    # Use tempfile.mkstemp to ensure proper handling of temporary files
    fd, temp_audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)  # Close the file descriptor
    tts.save(temp_audio_path)
    return temp_audio_path

# Create a video with audio and text
def create_video_with_script_audio(script_text, audio_file_path):
    """Generates a video with the given script text and audio."""
    # Set video parameters
    clip_duration = 10  # Duration for each screen showing the script text
    screen_width = 1920
    screen_height = 1080

    # Generate the video clip from script text
    clips = []
    for i in range(0, len(script_text), 100):  # Show a portion of text every 100 characters
        text = script_text[i:i+100]
        # Use 'DejaVuSans' font as a simple default font
        txt_clip = TextClip(
            text, 
            fontsize=50, 
            color='white', 
            bg_color='black', 
            size=(screen_width, screen_height),
            font="DejaVuSans"  # Use a basic font available in most environments
        )
        txt_clip = txt_clip.set_duration(clip_duration)
        clips.append(txt_clip)

    # Concatenate clips to form the video
    video = concatenate_videoclips(clips, method="compose")

    # Load the audio
    audio = AudioFileClip(audio_file_path)
    video = video.set_audio(audio)

    # Set final video output
    output_file = "/tmp/movie_video.mp4"
    video.write_videofile(output_file, fps=24)

    return output_file

# Streamlit App
st.title("Decentralized Autonomous Movie Creation System")

# Section 1: Script Generation
st.header("AI-Powered Script Generator")
prompt = st.text_area("Enter a prompt for the movie script:")
max_length = st.slider("Select the script length (tokens):", min_value=300, max_value=1000, value=600)

if st.button("Generate Script"):
    script_snippet = generate_script(prompt, max_length=max_length)
    if script_snippet == "Please provide a valid prompt.":
        st.warning(script_snippet)
    else:
        st.subheader("Generated Script:")
        st.write(script_snippet)

        # Extract skills/roles from the generated script
        extracted_skills = extract_skills_from_script(script_snippet)
        if extracted_skills:
            st.subheader("Extracted Skills/Roles:")
            st.write(", ".join(extracted_skills))
        else:
            st.warning("No relevant skills or roles found in the script.")

        # Convert script to audio
        st.subheader("Audio Version of the Script:")
        audio_file = text_to_audio(script_snippet)

        # Create video with the script audio
        video_file = create_video_with_script_audio(script_snippet, audio_file)

        # Provide download link for the video file
        with open(video_file, "rb") as video:
            st.download_button("Download Video", video, file_name="movie_script_video.mp4", mime="video/mp4")

        # Clean up the temporary files
        os.remove(audio_file)
        os.remove(video_file)
