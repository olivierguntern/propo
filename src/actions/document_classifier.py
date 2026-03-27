"""
Classification des types de documents.

STRATÉGIE HYBRIDE :
- Extension de fichier + mots-clés dans le nom → règle déterministe
- Contenu extrait → règle sur mots-clés comptables
- Fallback LLM uniquement si vraiment ambigu

PAS de LLM si le nom du fichier suffit.
"""
from pathlib import Path


_FILENAME_RULES: dict[str, list[str]] = {
    "facture": ["facture", "invoice", "fact_", "inv_"],
    "devis": ["devis", "quote", "offre", "proposition"],
    "releve_bancaire": ["releve", "statement", "rib", "bilan_banque"],
    "contrat": ["contrat", "contract", "cgv", "cgv_"],
    "bulletin_salaire": ["salaire", "bulletin", "fiche_paie", "paie"],
    "bilan": ["bilan", "liasse", "compte_resultat"],
}


def classify_document(filename: str, content_text: str = "") -> str:
    """Classifie un document à partir de son nom et/ou contenu."""
    stem = Path(filename).stem.lower()

    # Règle 1 : nom de fichier
    for doc_type, keywords in _FILENAME_RULES.items():
        if any(kw in stem for kw in keywords):
            return doc_type

    # Règle 2 : contenu textuel (si disponible)
    if content_text:
        text_lower = content_text.lower()
        scores: dict[str, int] = {}
        content_keywords = {
            "facture": ["total ttc", "tva", "numéro de facture", "date d'échéance"],
            "devis": ["devis n°", "validité", "estimation", "offre de prix"],
            "releve_bancaire": ["solde", "crédit", "débit", "iban"],
            "contrat": ["signataire", "clause", "résiliation", "durée du contrat"],
        }
        for doc_type, kws in content_keywords.items():
            scores[doc_type] = sum(1 for kw in kws if kw in text_lower)
        best = max(scores, key=scores.__getitem__)
        if scores[best] >= 2:
            return best

    return "autre"
