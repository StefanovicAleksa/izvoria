# Canvas: glossary_miner_queries_v4
# Path: app/features/glossary_miner/data/queries.py

# ==========================================
# SPARQL QUERY TEMPLATES
# ==========================================

GET_DOMAIN_TERMS_QUERY = """
SELECT DISTINCT ?item ?itemLabel ?alias ?aliasLang WHERE {{
  {{
    # Branch 1: Get the subclasses themselves (e.g., 'Bond', 'Derivative')
    ?item wdt:P279+ wd:{parent_id} .
  }}
  UNION
  {{
    # Branch 2: Get specific instances of the domain or its subclasses
    ?item wdt:P31/wdt:P279* wd:{parent_id} .
  }}

  # Ensure we only fetch items that actually have an English label
  ?item rdfs:label ?itemLabel .
  FILTER (LANG(?itemLabel) = "en")

  # Optionally fetch synonyms
  OPTIONAL {{
    ?item skos:altLabel ?alias .
    BIND(LANG(?alias) AS ?aliasLang)
    FILTER (?aliasLang IN ("en", "la"))
  }}
}}
LIMIT {limit}
"""