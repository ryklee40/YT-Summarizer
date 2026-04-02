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

def get_transcript(video_url, lang='en', cookies_path=None):
    """Fetch transcript using yt-dlp with optional cookie support."""
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': [lang, 'en'],
        'quiet': True,
        'no_warnings': True,
    }
    if cookies_path:
        ydl_opts['cookiefile'] = cookies_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            subtitles = info.get('subtitles') or {}
            auto_captions = info.get('automatic_captions') or {}

            # Prefer manual subtitles, fall back to auto captions
            all_subs = subtitles if subtitles else auto_captions

            if not all_subs:
                raise Exception("No subtitles or captions found for this video.")

            # Pick the first available English variant
            caption_url = None
            for key in [lang, 'en']:
                if key in all_subs:
                    formats = all_subs[key]
                    # Prefer json3, then any format
                    for fmt in formats:
                        if fmt.get('ext') == 'json3':
                            caption_url = fmt['url']
                            break
                    if not caption_url:
                        caption_url = formats[0]['url']
                    break

            if not caption_url:
                raise Exception("No English captions found.")

            headers = {'User-Agent': 'Mozilla/5.0'}
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
            except Exception:
                text = re.sub(r'<[^>]+>', '', response.text)
                text = text.replace('&amp;', '&').replace('&#39;', "'").replace('&quot;', '"')
                return ' '.join(text.split()).strip()

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