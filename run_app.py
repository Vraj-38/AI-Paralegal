import subprocess
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor

def run_fastapi():
    """Run the FastAPI backend server"""
    print("Starting FastAPI backend server...")
    return subprocess.Popen(
        [sys.executable, "-c", "import uvicorn; import main; uvicorn.run(main.app, host='0.0.0.0', port=8050)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

def run_streamlit():
    """Run the Streamlit frontend"""
    print("Starting Streamlit frontend...")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

def stream_output(process, prefix):
    """Stream the output of a subprocess with a prefix"""
    for line in iter(process.stdout.readline, ""):
        if line:
            print(f"{prefix}: {line.strip()}")
        else:
            break

def main():
    # Start both processes
    fastapi_process = run_fastapi()
    # Wait a bit for FastAPI to start before launching Streamlit
    time.sleep(3)
    streamlit_process = run_streamlit()
    
    # Stream their output in separate threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(stream_output, fastapi_process, "FastAPI")
        executor.submit(stream_output, streamlit_process, "Streamlit")
    
    try:
        # Keep the main process running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        # Terminate both processes
        fastapi_process.terminate()
        streamlit_process.terminate()
        print("Servers shut down successfully!")

if __name__ == "__main__":
    main() 