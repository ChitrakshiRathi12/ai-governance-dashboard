"""
app_hf.py
HuggingFace Spaces entry point.
Runs the Streamlit dashboard on HuggingFace Spaces.
Rename to app.py when deploying to HF Spaces.
"""

import subprocess
import sys
import os

# HuggingFace Spaces uses port 7860 by default
os.system(
    "streamlit run dashboard/streamlit_app.py "
    "--server.port 7860 "
    "--server.address 0.0.0.0 "
    "--server.headless true "
    "--server.fileWatcherType none"
)
