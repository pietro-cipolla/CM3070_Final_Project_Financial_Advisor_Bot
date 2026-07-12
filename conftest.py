"""
Ensures the project root (this file's directory) is on sys.path so that
`import src...` works regardless of the directory pytest is invoked from.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The rag_pipeline / advisor modules instantiate an OpenAI client at import
# time using OPENAI_API_KEY. Tests never make real API calls (the client is
# mocked), but the SDK requires a non-empty key just to construct the
# client, so we provide a dummy one if the environment doesn't have it.
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key-for-pytest")
