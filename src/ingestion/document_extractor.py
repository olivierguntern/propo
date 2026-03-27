"""
Extraction de texte depuis les pièces jointes.
Supporte PDF (PyPDF2). Extensible pour Word, Excel.

CHOIX : on n'utilise PAS de LLM pour extraire le texte brut,
c'est une opération déterministe.
"""
import io
from pathlib import Path

try:
    from PyPDF2 import PdfReader
    _PYPDF2_AVAILABLE = True
except ImportError:
    _PYPDF2_AVAILABLE = False


def extract_text_from_pdf(content: bytes) -> str:
    """Extrait le texte d'un PDF. Retourne chaîne vide si échec."""
    if not _PYPDF2_AVAILABLE:
        return "[PyPDF2 non installé - extraction impossible]"
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as exc:
        return f"[Erreur extraction PDF : {exc}]"


def extract_text_from_file(filename: str, content: bytes) -> str:
    """Dispatch selon l'extension du fichier."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(content)
    # Extensible : .docx, .xlsx, etc.
    return content.decode("utf-8", errors="replace")
