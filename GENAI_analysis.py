import base64
import os
import time
from google import genai
from google.genai import types
import markdown


def markdown_to_text(md_text):
    """Converts markdown text to plain text."""
    return markdown.markdown(md_text).replace("<p>", "").replace("</p>", "").replace("<br>", "\n")


def generate():
    #Create a new .env file and obtail the API key from Google AI Studio and paste it there or you can directly paste here too.
    api_key = "Replace with your key"
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Set it in environment variables.")

    client = genai.Client(api_key=api_key)

    video_file = "basket_detection.mp4"
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"File {video_file} not found.")

    # Upload file
    uploaded_file = client.files.upload(file=video_file)
    file_id = uploaded_file.name  # Get file ID

    # Wait for the file to become ACTIVE
    max_attempts = 10  # Maximum retries
    wait_time = 20  # Wait time in seconds per attempt

    for attempt in range(max_attempts):
        file_info = client.files.get(name=file_id)
        if file_info.state == "ACTIVE":
            print(f"File {file_id} is now ACTIVE. Proceeding with generation.")
            break
        else:
            print(
                f"Attempt {attempt + 1}/{max_attempts}: File {file_id} is in {file_info.state} state. Waiting...")
            time.sleep(wait_time)
    else:
        raise RuntimeError(
            f"File {file_id} did not become ACTIVE after {max_attempts * wait_time} seconds.")

    # Prepare prompt
    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_uri(
                file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type)]
        ),
        types.Content(
            role="user",
            parts=[types.Part.from_text(
                text="Summarize this video and provide customer analysis and insights for my retail shop. "
                     "Ensure the response is within 500 words."
            )]
        ),
    ]

    # Generation config
    generate_content_config = types.GenerateContentConfig(
        temperature=0.8,
        top_p=0.9,
        top_k=40,
        max_output_tokens=1024,  # 8192 is excessive for a short summary
        response_mime_type="text/plain",
    )

    # Generate content
    response_text = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
        response_text += chunk.text

    return markdown_to_text(response_text)  # Ensure it's plain text

    


# ✅ Prevent auto-execution when imported
if __name__ == "__main__":
    print(generate())  # Runs only if executed directly
