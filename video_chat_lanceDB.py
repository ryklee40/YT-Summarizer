import yt_dlp
import requests
import re
import lancedb
from sentence_transformers import SentenceTransformer

# -----------------------------
# STEP 1: Get transcript
# -----------------------------
def get_transcript(video_url, lang='en'):
    """
    Fetch English subtitles or auto-generated captions using yt-dlp.
    """
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': [lang],
        'quiet': True,
        'no_warnings': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        subtitles = info.get('subtitles') or info.get('automatic_captions')

        if not subtitles:
            raise Exception("No subtitles found.")

        subtitle_key = None
        print('Finding Key')
        for key, formats in subtitles.items():
            if lang in formats[0].get('name', '').lower():
                subtitle_key = key
                break
        if not subtitle_key:
            raise Exception(f"No '{lang}' captions found.")
        print('Key found')
        caption_url = subtitles[subtitle_key][0]['url']
        response = requests.get(caption_url)
        
        # Try to parse as JSON (json3 format)
        try:
            data = response.json()
            text_lines = []
            for event in data.get('events', []):
                for seg in event.get('segs', []):
                    text = seg.get('utf8', '').strip()
                    if text and text != '\n':
                        text_lines.append(text)
            return ' '.join(text_lines)
        except:
            # Fall back to XML/plain text
            text = re.sub(r'<[^>]+>', '', response.text)
            text = text.replace('&amp;', '&').replace('&#39;', "'").replace('&quot;', '"')
            text = ' '.join(text.split())
            return text.strip()

# -----------------------------
# STEP 2: Split transcript into chunks
# -----------------------------
def chunk_text(text, lines_per_chunk=4):
    """Split text by common delimiters (periods, question marks, music notes, newlines, etc.)."""
    # Split on periods, question marks, exclamation marks, music notes, and newlines
    lines = re.split(r'[.!?♪\n]+', text)
    
    # Clean up - remove empty strings and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]
    
    # Group into chunks
    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk = ' '.join(lines[i:i + lines_per_chunk])
        if chunk:
            chunks.append(chunk)
    
    return chunks if chunks else [text]


# -----------------------------
# STEP 3: Build embeddings + store in LanceDB
# -----------------------------
def build_vector_store(transcript_text, db_path="./lancedb", table_name="transcript"):
    chunks = list(chunk_text(transcript_text))
    print(f"Split transcript into {len(chunks)} chunks.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)

    db = lancedb.connect(db_path)
    
    # Create data list with chunks and embeddings
    data = [
        {"text": chunks[i], "vector": embeddings[i].tolist()}
        for i in range(len(chunks))
    ]
    
    # Create table with data directly
    table = db.create_table(table_name, data=data, mode="overwrite")

    print("✅ Transcript embedded and stored in LanceDB.")
    return model, table


# -----------------------------
# STEP 4: Ask questions
# -----------------------------
def ask_question(model, table):
    while True:
        query = input("\nAsk a question (or 'quit'): ").strip()
        if query.lower() == "quit":
            break

        query_embedding = model.encode([query])[0]
        results = table.search(query_embedding).limit(3).to_df()

        print("\n🔍 Top relevant transcript snippets:\n")
        for text in results["text"]:
            print("•", text[:300], "...\n")


def main():
    video_url = input("🎥 Enter YouTube video URL: ").strip()

    try:
        print("\n📜 Fetching transcript...")
        transcript = get_transcript(video_url)

        print("\n🧮 Building local LanceDB store...")
        model, table = build_vector_store(transcript)

        print("\n💬 Ready to chat with the video!")
        ask_question(model, table)  
  
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
