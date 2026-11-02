import subprocess
import time
import urllib.request
import sys

proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print("Starting Streamlit app process...")
time.sleep(5)

try:
    with urllib.request.urlopen("http://127.0.0.1:8501", timeout=5) as response:
        status = response.getcode()
        print(f"App probe succeeded with HTTP status code: {status}")
except Exception as e:
    print(f"Probe error: {e}")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
    print("Streamlit process stopped cleanly.")

