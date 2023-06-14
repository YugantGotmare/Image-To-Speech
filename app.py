from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from gtts import gTTS
import streamlit as st
from PIL import Image
import io

load_dotenv(find_dotenv())


def img2text(image):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(image)[0]["generated_text"]
    return text


def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    audio_file = "output.mp3"
    tts.save(audio_file)
    return audio_file


def main():
    st.title("Image to Speech")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image from the BytesIO object
        image = Image.open(io.BytesIO(uploaded_file.read()))

        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to text
        text = img2text(image)

        # Convert text to speech
        audio_file = text_to_speech(text)

        # Provide audio controls to play the speech
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")

        # Display the extracted text
        st.subheader("Extracted Text:")
        st.write(text)


if __name__ == "__main__":
    main()
