# Companion example, not printed in the book; uses lst-1.8
from pathlib import Path
import runpy

from pydantic import ValidationError


listing = Path(__file__).resolve().parents[1] / "chapters/ch01/lst-1.8.py"
ChatbotInput = runpy.run_path(str(listing))["ChatbotInput"]

accepted = ChatbotInput(user_message="Hello", context_limit=2000)
print(f"Accepted: {accepted}")

try:
    ChatbotInput(user_message="", context_limit=2000)
except ValidationError as error:
    print("Invalid input:")
    print(error)
