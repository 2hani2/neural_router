#!/usr/bin/env python3
# run.py — starts FastAPI + ngrok tunnel in one command
# Usage: python run.py

import os, sys, time, threading
from dotenv import load_dotenv

load_dotenv()

def start_ngrok(port: int):
    """Start ngrok tunnel and print the public URL."""
    try:
        from pyngrok import ngrok, conf
        # If you have a free ngrok account, set NGROK_AUTH_TOKEN in .env
        token = os.getenv("NGROK_AUTH_TOKEN")
        if token:
            ngrok.set_auth_token(token)
        time.sleep(1.5)  # wait for uvicorn to start
        tunnel = ngrok.connect(port, "http")
        print("\n" + "="*55)
        print(f"  🌐 Public HTTPS URL: {tunnel.public_url}")
        print(f"  📡 Local:           http://localhost:{port}")
        print("="*55 + "\n")
    except ImportError:
        print("[ngrok] pyngrok not installed — run: pip install pyngrok")
    except Exception as e:
        print(f"[ngrok] Failed to start tunnel: {e}")
        print(f"[ngrok] App still running at http://localhost:{port}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))

    # Validate required env vars
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    if not os.path.exists(os.getenv("MODEL_PATH", "./model/nanoqa_v4_best.pt")):
        print(f"ERROR: Model not found at {os.getenv('MODEL_PATH', './model/nanoqa_v4_best.pt')}")
        print("Put nanoqa_v4_best.pt in the ./model/ folder")
        sys.exit(1)

    print("Starting Neural Router v4...")
    print(f"Model : {os.getenv('MODEL_PATH')}")
    print(f"Port  : {port}")

    # Start ngrok in background thread
    t = threading.Thread(target=start_ngrok, args=(port,), daemon=True)
    t.start()

    # Start FastAPI
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
