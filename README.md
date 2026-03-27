# Agent d'analyse et traitement automatisé des demandes clients
### Niveau Dougs — Agent IA pour comptables

---

## Contexte

Un comptable chez Dougs reçoit quotidiennement des emails clients avec des factures, devis, questions comptables et pièces jointes. L'objectif est d'**automatiser la lecture, la classification, et la génération de réponses**, tout en maintenant une supervision humaine sur les cas sensibles.

---

## Architecture

```
Email entrant
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  1. INGESTION                                           │
│  src/ingestion/                                         │
│  • Lecture email (mock JSON → prod: IMAP/API)           │
│  • Extraction texte PDF (PyPDF2) — PAS de LLM           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  2. ANALYSE                                             │
│  src/analysis/                                          │
│                                                         │
│  classifier.py                                          │
│  ├── Règles mots-clés (rapide, déterministe, 40-60%)    │
│  └── LLM si ambigu → JSON structuré + confidence        │
│                                                         │
│  extractor.py  [LLM]                                    │
│  └── Extraction montant, date, client, urgence          │
│                                                         │
│  validator.py  [PAS DE LLM]                             │
│  └── TVA valide ? Montant > 0 ? Date dans le passé ?    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  3. DÉCISION D'ESCALADE  [PAS DE LLM]                   │
│  src/actions/escalation.py                              │
│  • confidence < 70% → humain                            │
│  • erreurs de validation → humain                       │
│  • montant facture > 10 000 EUR → humain                │
│  • urgence haute → humain                               │
└────────────────────────┬────────────────────────────────┘
                         │
               ┌─────────┴──────────┐
               │ Escalade           │ Normal
               ▼                   ▼
          Notif humain    ┌─────────────────────┐
                          │  4. RAG             │
                          │  src/rag/           │
                          │  ChromaDB           │
                          │  • Règles comptables│
                          │  • FAQ              │
                          │  • Templates        │
                          └─────────┬───────────┘
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │  5. ACTION  [LLM]   │
                          │  src/actions/       │
                          │  • Réponse email    │
                          │  • Classif document │
                          └─────────┬───────────┘
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │  6. MONITORING      │
                          │  Logs structurés    │
                          │  Métriques          │
                          └─────────────────────┘
```

---

## Où utilise-t-on le LLM ? Où pas ?

| Étape | LLM ? | Pourquoi |
|-------|-------|----------|
| Extraction texte PDF | Non | Opération déterministe (PyPDF2) |
| Classification (mots-clés évidents) | Non | Règles rapides, auditables, coûts réduits |
| Classification (cas ambigus) | Oui | Compréhension du langage naturel |
| Extraction d'informations | Oui | Lecture de contexte complexe |
| Validation TVA / montants / dates | Non | Règles comptables déterministes — hallucination = risque financier |
| Logique d'escalade | Non | Décision binaire basée sur des seuils |
| Classification de documents | Non | Règles sur noms de fichiers + mots-clés |
| Génération de réponse | Oui | Personnalisation du langage naturel |

---

## Gestion des erreurs et hallucinations

```
LLM renvoie JSON malformé
    → regex extraction + fallback Classification(category="autre", confidence=0)
    → escalade automatique vers humain

Confidence LLM < 70%
    → escalade automatique (seuil configurable via CONFIDENCE_THRESHOLD)

Données financières invalides (TVA incorrecte, montant négatif)
    → blocage avant action + escalade

Facture > 10 000 EUR
    → validation humaine obligatoire
```

---

## Structure du projet

```
propo/
├── src/
│   ├── agent.py                   # Orchestrateur principal
│   ├── models.py                  # Modèles Pydantic (validation)
│   ├── ingestion/
│   │   ├── email_reader.py        # Lecture emails
│   │   └── document_extractor.py # Extraction PDF
│   ├── analysis/
│   │   ├── classifier.py          # Classification (règles + mots-clés + LLM)
│   │   ├── extractor.py           # Extraction infos (LLM)
│   │   └── validator.py           # Validation métier (PAS LLM)
│   ├── rag/
│   │   ├── vector_store.py        # ChromaDB
│   │   └── knowledge_base.py     # Chargement base de connaissances
│   ├── actions/
│   │   ├── response_generator.py # Génération réponse (LLM + RAG)
│   │   ├── document_classifier.py # Classification document
│   │   └── escalation.py         # Logique d'escalade
│   └── monitoring/
│       └── logger.py              # Logs structurés + métriques
├── data/
│   ├── mock_emails/emails.json    # 5 emails de test
│   └── knowledge_base/            # Règles comptables, FAQ, templates
├── tests/
│   └── test_agent.py              # Tests unitaires (sans LLM)
├── demo.py                        # Script de démonstration
├── requirements.txt
└── .env.example
```

---

## Installation et lancement

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Configurer l'API key
cp .env.example .env
# Éditer .env et renseigner ANTHROPIC_API_KEY

# 3. Lancer la démo (5 emails mock)
python demo.py

# 4. Lancer les tests unitaires (pas d'API key nécessaire)
pytest tests/ -v
```

---

## Choix techniques

| Composant | Choix | Raison |
|-----------|-------|--------|
| LLM | Claude (Anthropic) | Meilleure compréhension du français, JSON fiable |
| Vector DB | ChromaDB | Simple, local, pas d'infra externe pour le MVP |
| Embeddings | sentence-transformers (multilingue) | Français natif, léger, open-source |
| Validation | Pydantic v2 | Schémas stricts, erreurs lisibles |
| Logs | JSON structuré | Observable en prod (Datadog, ELK) |

---

## Évolutions possibles

- **Connexion IMAP/Gmail API** : remplacer le mock email_reader
- **Base vectorielle managée** : Pinecone ou Weaviate en prod
- **Interface de révision** : UI pour les cas escaladés
- **Feedback loop** : corrections humaines → réentraînement des règles
- **Détection de doublons** : éviter de traiter deux fois la même facture
