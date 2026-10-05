import os
import tempfile

import streamlit as st
from resume_processor import load_resume, analyze_resume, store_to_vectorstore, run_self_query, GEMINI_MODEL


def friendly_error(e):
    msg = str(e)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return (f"Daily free quota for '{GEMINI_MODEL}' is used up. "
                "Set GEMINI_MODEL in .env to another model (e.g. gemini-3.5-flash), "
                "wait for the quota to reset, or enable billing.")
    if "UNAVAILABLE" in msg or "503" in msg:
        return f"'{GEMINI_MODEL}' is overloaded right now. Try again in a minute."
    return msg

st.set_page_config(page_title="AI Resume Screener")
st.title("AI Resume Screener")
st.markdown("Upload a resume and analyze it using AI. Then run smart searches over previous resumes.")

job_desc = st.text_area("Paste Job Description")
uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if st.button("Analyze & Store"):
    if not uploaded_file or not job_desc:
        st.warning("Please paste a job description and upload a resume first.")
    else:
        # Save the upload to a temp file so the loaders can read it from disk
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("Analyzing & Storing Resume..."):
                docs = load_resume(tmp_path)
                st.session_state.report = analyze_resume(docs, job_desc)  # uses Gemini inside resume_processor.py
                store_to_vectorstore(docs, uploaded_file.name)
            st.success("Analysis complete and stored!")
        except Exception as e:
            st.error(f"Gemini request failed: {friendly_error(e)}")
        finally:
            os.remove(tmp_path)

# Kept in session_state so it survives Streamlit reruns (e.g. clicking Download or Search)
if "report" in st.session_state:
    st.subheader("AI Resume Summary")
    st.write(st.session_state.report)
    st.download_button("Download Report", st.session_state.report, file_name="resume_analysis.txt")

st.divider()
st.subheader("Ask Anything About Stored Resumes")
query = st.text_input("Type your smart query here (e.g., 'Python developer with AWS')")

if st.button("Search Resumes") and query:
    with st.spinner("Searching..."):
        try:
            results = run_self_query(query)
        except Exception as e:
            st.error(f"Search failed: {friendly_error(e)}")
            st.stop()
        if results:
            for i, res in enumerate(results, 1):
                st.markdown(f"**Result {i}:** _{res.metadata.get('source', 'unknown')}_")
                st.write(res.page_content.strip())
        else:
            st.warning("No matches found.")
