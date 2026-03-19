# Canvas: glossary_miner_queries_v2
# Path: app/features/glossary_miner/data/queries.py

# ==========================================
# SPARQL QUERY TEMPLATES
# ==========================================

GET_DOMAIN_TERMS_QUERY = """
SELECT ?item ?itemLabel ?alias ?aliasLang WHERE {{
  # 1. Find all items that are an instance of or subclass of the target domain
  ?item wdt:P31/wdt:P279* wd:{parent_id} .

  # 2. Get the primary English label
  ?item rdfs:label ?itemLabel .
  FILTER (LANG(?itemLabel) = "en")

  # 3. Get the aliases (synonyms) optionally
  OPTIONAL {{
    ?item skos:altLabel ?alias .
    BIND(LANG(?alias) AS ?aliasLang)

    # THE MAGIC: Only keep English or Latin aliases!
    FILTER (?aliasLang IN ("en", "la"))
  }}
}}
LIMIT {limit}
"""