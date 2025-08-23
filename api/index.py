# Import the Flask WSGI app from server.py and expose it as `app` for Vercel
from server import app

# Vercel detects a WSGI application if an `app` variable is exported in api/*.py
