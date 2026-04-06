import sys
import os

# Redirect basic executions directly to the new frontend
from frontend.app import demo
from src.config import settings

if __name__ == "__main__":
    print("Launching frontend application (Scalable format enabled...)")
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.PORT,
        share=False,
    )
