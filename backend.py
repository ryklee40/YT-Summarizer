import yt_dlp
import requests
import re
import lancedb
from sentence_transformers import SentenceTransformer

"""
def get_transcript(video_url, lang='en'):
    #Fetch English subtitles or auto-generated captions using yt-dlp.
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
        for key, formats in subtitles.items():
            if lang in formats[0].get('name', '').lower():
                subtitle_key = key
                break
        
        if not subtitle_key:
            raise Exception(f"No '{lang}' captions found.")
        
        caption_url = subtitles[subtitle_key][0]['url']
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(caption_url, headers=headers)
        
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
            text = re.sub(r'<[^>]+>', '', response.text)
            text = text.replace('&amp;', '&').replace('&#39;', "'").replace('&quot;', '"')
            text = ' '.join(text.split())
            return text.strip()
"""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

def get_transcript(video_url, lang='en'):
    """Fetch transcript text using youtube_transcript_api with .list() and .fetch()."""
    # Extract YouTube video ID
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    if not match:
        raise Exception("Invalid YouTube URL.")
    video_id = match.group(1)

    try:
        # Create an instance and call list() on it
        api = YouTubeTranscriptApi()
        transcripts = api.list(video_id)
        
        # Try to find a transcript in the requested language
        try:
            transcript_obj = transcripts.find_transcript([lang])
        except NoTranscriptFound:
            # Fallback to English
            try:
                transcript_obj = transcripts.find_manually_created_transcript(['en'])
            except NoTranscriptFound:
                transcript_obj = transcripts.find_generated_transcript(['en'])
        
        # Fetch transcript contents
        transcript_data = transcript_obj.fetch()
        
        # Access text attribute directly (not using .get())
        text = " ".join(
            [entry.text.strip() for entry in transcript_data if hasattr(entry, 'text') and entry.text.strip()]
        )
        return text.strip()

    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise Exception(f"No transcript found for language '{lang}'.")
    except Exception as e:
        raise Exception(f"Could not fetch transcript: {e}")

def chunk_text(text, lines_per_chunk=4):
    """Split text by common delimiters (periods, question marks, music notes, newlines, etc.)."""
    lines = re.split(r'[.!?♪\n]+', text)
    lines = [line.strip() for line in lines if line.strip()]
    
    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk = ' '.join(lines[i:i + lines_per_chunk])
        if chunk:
            chunks.append(chunk)
    
    return chunks if chunks else [text]


def build_vector_store(transcript_text, db_path="./lancedb", table_name="transcript"):
    """Build embeddings and store in LanceDB."""
    chunks = chunk_text(transcript_text)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=False)

    db = lancedb.connect(db_path)
    
    data = [
        {"text": chunks[i], "vector": embeddings[i].tolist()}
        for i in range(len(chunks))
    ]
    
    table = db.create_table(table_name, data=data, mode="overwrite")
    
    return model, table, len(chunks)


def search_transcript(model, table, query, limit=3):
    """Search for relevant transcript snippets."""
    query_embedding = model.encode([query])[0]
    results = table.search(query_embedding.tolist()).limit(limit).to_pandas()
    return results