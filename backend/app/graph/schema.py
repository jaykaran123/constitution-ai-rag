"""Knowledge graph schema: node labels, relationship types and integrity rules."""
from __future__ import annotations

GRAPH_SCHEMA_VERSION = 2

CONSTITUTIONAL_LABELS = ["Constitution", "Preamble", "Part", "Chapter", "Article", "Clause", "SubClause",
                         "Schedule", "Amendment"]
LEGAL_LABELS = ["FundamentalRight", "ConstitutionalPrinciple", "DirectivePrinciple", "FundamentalDuty",
                "LegalConcept", "Statute", "StatutorySection", "Offence", "LegalRemedy", "GovernmentAuthority",
                "Court", "CaseLaw", "LegalRestriction", "Exception"]
PROVENANCE_LABELS = ["Document", "DocumentVersion", "Chunk", "Source", "Citation"]
SCENARIO_LABELS = ["LegalScenario", "Incident", "LegalIssue", "UserRight"]
ALL_LABELS = CONSTITUTIONAL_LABELS + LEGAL_LABELS + PROVENANCE_LABELS + SCENARIO_LABELS

# Allowed (source label, relationship, target label) triples. Anything else is rejected.
RELATIONSHIP_RULES: dict[str, set[tuple[str, str]]] = {
    "CONTAINS": {("Constitution", "Preamble"), ("Constitution", "Part"), ("Constitution", "Schedule"),
                 ("Part", "Chapter"), ("Part", "Article"), ("Chapter", "Article"), ("Statute", "StatutorySection"),
                 ("Document", "Chunk")},
    "HAS_VERSION": {("Document", "DocumentVersion")},
    "TEXT_OF": {("DocumentVersion", "Constitution")},
    "PUBLISHED_BY": {("DocumentVersion", "Source"), ("Statute", "Source")},
    "HAS_CLAUSE": {("Article", "Clause")},
    "HAS_SUBCLAUSE": {("Clause", "SubClause"), ("Article", "SubClause")},
    "PROTECTS": {("Article", "FundamentalRight")},
    "DECLARES": {("Preamble", "ConstitutionalPrinciple")},
    "DEFINES": {("Article", "LegalConcept"), ("Clause", "LegalConcept"), ("StatutorySection", "Offence"),
                ("Article", "Offence"), ("Clause", "Offence")},
    "SUBJECT_TO": {("Article", "LegalRestriction"), ("Clause", "LegalRestriction"),
                   ("SubClause", "LegalRestriction")},
    "LIMITED_BY": {("Article", "Exception"), ("Clause", "Exception"), ("SubClause", "Exception")},
    "STATED_IN": {("LegalRestriction", "Clause"), ("Exception", "Clause"), ("Exception", "Article"),
                  ("LegalRestriction", "Article")},
    "REFERENCES": {("Article", "Article"), ("Citation", "Source")},
    "RELATED_TO": {("Article", "Article"), ("LegalConcept", "Article"), ("LegalConcept", "Clause")},
    "INTERPRETED_BY": {("Article", "CaseLaw")},
    "APPLIES_TO": {("CaseLaw", "LegalConcept")},
    "AMENDED": {("Amendment", "Article"), ("Amendment", "Clause"), ("Amendment", "SubClause"),
                ("Amendment", "Part"), ("Amendment", "Chapter"), ("Amendment", "Preamble"),
                ("Amendment", "Schedule")},
    "INVOLVES": {("LegalScenario", "LegalIssue")},
    "MAY_ENGAGE": {("LegalIssue", "Article"), ("LegalIssue", "Clause"), ("LegalIssue", "SubClause")},
    "MAY_INVOLVE": {("LegalIssue", "Offence")},
    "AVAILABLE_FOR": {("LegalRemedy", "LegalIssue")},
    "HANDLES": {("GovernmentAuthority", "LegalRemedy"), ("Court", "LegalRemedy")},
    "PROVIDED_BY": {("LegalRemedy", "StatutorySection"), ("LegalRemedy", "Clause"), ("LegalRemedy", "Article")},
    "GUARANTEED_BY": {("UserRight", "Clause"), ("UserRight", "Article"), ("UserRight", "SubClause")},
    "SUPPORTS": {("Chunk", "Article"), ("Chunk", "Clause"), ("Chunk", "Preamble"), ("Chunk", "Schedule")},
    "CITED_AS": {("StatutorySection", "Citation")},
    "ESTABLISHED_BY": {("Court", "Article"), ("Court", "Clause")},
}

# Relationships every node of a label is expected to have (used by the inspection utility).
REQUIRED_RELATIONSHIPS = {
    "Article": [("in", "CONTAINS"), ("in", "SUPPORTS")],
    "Clause": [("in", "HAS_CLAUSE")],
    "SubClause": [("in", "HAS_SUBCLAUSE")],
    "Chunk": [("in", "CONTAINS"), ("out", "SUPPORTS")],
    "StatutorySection": [("in", "CONTAINS")],
    "LegalIssue": [("out", "MAY_ENGAGE")],
}

VERIFICATION_STATUSES = {"verified", "curated", "unverified"}
EXTRACTION_METHODS = {"structural", "text_pattern", "curated", "footnote", "model"}


def node_key(label: str, ident: str) -> str:
    return f"{label}:{ident}"


def provision_label_to_node(kind: str) -> str | None:
    return {"article": "Article", "clause": "Clause", "subclause": "SubClause", "part": "Part",
            "chapter": "Chapter", "preamble": "Preamble", "schedule": "Schedule"}.get(kind)
