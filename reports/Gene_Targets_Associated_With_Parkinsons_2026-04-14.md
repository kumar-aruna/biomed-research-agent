# Research report

**Query:** What gene targets are associated with Parkinson's disease and what does recent research say about them?

**Date:** April 14, 2026 at 16:25

**Runtime:** 19.4s

---

  Step 1: planner → ['disease_lookup']
           ↳ {'data': {'search': {'hits': [{'id': 'MONDO_0005180', 'name': 'Parkinson disease', 'description': 'A progressive degener...
  Step 2: planner → ['disease_evidence']
           ↳ {'total_targets': 7439, 'top_10': [{'gene': 'LRRK2', 'ensembl_id': 'ENSG00000188906', 'score': 0.8802}, {'gene': 'SNCA',...
  Step 3: planner → ['fetch_articles']
           ↳ /Users/arunak/Documents/langgraph-learning/lit_workspace/catalog.json
  Step 4: planner → ['build_chunks']
           ↳ /Users/arunak/Documents/langgraph-learning/lit_workspace/chunks.jsonl

✅ Chunked 6 passages from 1 articles:
   📖 Full t...
  Step 5: planner → ['bm25_search']
           ↳ /Users/arunak/Documents/langgraph-learning/lit_workspace/bm25_evidence.txt

(Saved 6 passages — pass ONLY that path to l...
  Step 6: planner → ['protein_info', 'protein_info']
           ↳ {'protein': '', 'accession': '', 'gene': '', 'organism': '', 'function': ''}
           ↳ {'protein': '', 'accession': '', 'gene': '', 'organism': '', 'function': ''}
  Step 7: planner → ['web_search']
           ↳ [{'title': "Molecular Pathology and Neuronal Networks in LRRK2 Parkinson's Disease", 'body': "Clinicaldiagnosis ofParkin...

---

## Disease Overview
Parkinson's disease is a progressive degenerative disorder of the central nervous system characterized by the loss of dopamine-producing neurons in the substantia nigra and the presence of Lewy bodies. Symptoms include tremor, muscle rigidity, slowing of voluntary movements, a tendency to fall back, and a mask-like facial expression.

## Top Gene Targets
| Gene Symbol | Ensembl ID        | Association Score |
|-------------|-------------------|-------------------|
| LRRK2       | ENSG00000188906   | 0.8802            |
| SNCA        | ENSG00000145335   | 0.8748            |
| PINK1       | ENSG00000158828   | 0.8606            |
| PRKN        | ENSG00000185345   | 0.8574            |
| ATP13A2     | ENSG00000159363   | 0.8574            |
| DNAJC6      | ENSG00000116675   | 0.8250            |
| PARK7       | ENSG00000116288   | 0.8187            |
| FBXO7       | ENSG00000100225   | 0.8021            |
| VPS35       | ENSG00000069329   | 0.7857            |
| VPS13C      | ENSG00000129003   | 0.7552            |

## Literature Findings
Recent genetic screening in familial Parkinson's disease cases from Central and South America identified pathogenic variants in genes such as FBXO7, LRRK2, PARK7, PINK1, PRKN, and SNCA. These findings underscore the genetic diversity and complexity in Parkinson's disease (PMID: 39051491, abstract only).

## Additional Insights
Recent clinical trials have focused on LRRK2, a significant gene in Parkinson's pathology. The NEULARK study is testing an investigational treatment called NEU-411 to block LRRK2 activity, linked to neurodegeneration. Additionally, the first-in-human trial of LRRK2-targeting antisense oligonucleotide BIIB094 showed promising results in terms of safety and dose-dependent effects.

## Summary
Parkinson's disease is associated with several gene targets, including LRRK2, SNCA, and PINK1. Recent research highlights the genetic complexity of the disease and ongoing clinical trials targeting LRRK2 offer promising avenues for treatment development.