import streamlit as st
from backend import get_transcript, build_vector_store, search_transcript

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="YouTube Video Chat", page_icon="🎥", layout="wide")

st.title("🎥 YouTube Video Chat")
st.markdown("Ask questions about any YouTube video with captions!")

# ------------------------------------------------------------
# SIDEBAR — VIDEO INPUT
# ------------------------------------------------------------
with st.sidebar:
    st.header("Load Video")
    video_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Load Video", type="primary"):
        if video_url:
            try:
                with st.spinner("Fetching transcript..."):
                    transcript = get_transcript(video_url)
                
                with st.spinner("Building search index..."):
                    model, table, num_chunks = build_vector_store(transcript)
                
                # Store in session state
                st.session_state['model'] = model
                st.session_state['table'] = table
                st.session_state['transcript'] = transcript
                st.session_state['video_url'] = video_url
                
                st.success(f"✅ Video loaded! ({num_chunks} chunks)")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("Please enter a YouTube URL")

    # Show video info if loaded
    if 'video_url' in st.session_state:
        st.divider()
        st.markdown("**Current Video:**")
        st.video(st.session_state['video_url'])

# ------------------------------------------------------------
# MAIN CHAT INTERFACE
# ------------------------------------------------------------
if 'table' in st.session_state:
    st.header("💬 Ask Questions")
    
    query = st.text_input("Your question:", placeholder="What is this video about?")
    
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            results = search_transcript(st.session_state['model'], st.session_state['table'], query)
        
        st.subheader("🔍 Relevant Snippets:")
        
        for idx, row in results.iterrows():
            with st.expander(f"Result {idx+1}", expanded=(idx==0)):
                st.write(row['text'])
                if '_distance' in row:
                    st.caption(f"Relevance score: {row['_distance']:.4f}")
else:
    st.info("👈 Enter a YouTube URL in the sidebar to get started!")

# ------------------------------------------------------------
# TRANSCRIPT VIEWER & DOWNLOAD
# ------------------------------------------------------------
if 'transcript' in st.session_state:
    st.header("📜 Full Transcript")

    with st.expander("Show Transcript"):
        st.write(st.session_state['transcript'])

    st.download_button(
        label="💾 Download Transcript",
        data=st.session_state['transcript'],
        file_name="transcript.txt",
        mime="text/plain"
    )

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.divider()
st.caption("Built with Streamlit, LanceDB, and yt-dlp")

# Run using:
#   streamlit run app.py
