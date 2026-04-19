# Research report

**Query:** What gene targets are associated with Parkinson's disease and what does recent research say about them?

**Date:** April 13, 2026 at 14:00

**Runtime:** 19.6s

---

  Step 1: planner → ['disease_lookup']
           ↳ {'data': {'search': {'hits': [{'id': 'MONDO_0005180', 'name': 'Parkinson disease', 'description': 'A progressive degener...
  Step 2: planner → ['disease_evidence']
           ↳ {'total_targets': 7439, 'top_10': [{'gene': 'LRRK2', 'ensembl_id': 'ENSG00000188906', 'score': 0.8802}, {'gene': 'SNCA',...
  Step 3: planner → ['fetch_articles']
           ↳ /Users/arunak/Documents/langgraph-learning/lit_workspace/catalog.json
  Step 4: planner → ['build_chunks']
           ↳ /Users/arunak/Documents/langgraph-learning/lit_workspace/chunks.jsonl

✅ Chunked 111 passages from 19 articles:
   📖 Ful...
  Step 5: planner → ['bm25_search']
           ↳ /Users/arunak/Documents/langgraph-learning/lit_workspace/bm25_evidence.txt

(Saved 6 passages — pass ONLY that path to l...
  Step 6: planner → ['protein_info', 'protein_info', 'web_search']
           ↳ {'protein': '', 'accession': '', 'gene': '', 'organism': '', 'function': ''}
           ↳ {'protein': '', 'accession': '', 'gene': '', 'organism': '', 'function': ''}
           ↳ [{'title': "Gene therapy for Parkinson's disease: trials and technical advances", 'body': "Genetherapy has long held pro...

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
- **Postural Instability**: A meta-analysis reported postural instability in 31% of LRRK2, 29% of SNCA, 39% of PINK1, 40% of PRKN, and 34% of ATP13A2 patients. Progression-free survival from postural instability at 10 years was longest in ATP13A2 (97%) and shortest in SNCA (50%) (PMID: 32436106, abstract only).
- **Diagnostic Delay**: Studies have highlighted diagnostic delays in Parkinson's disease caused by mutations in PRKN, SNCA, DJ1, PINK1, LRRK2, GBA, and ATP13A2 (PMID: 30692050, abstract only).

## Additional Insights
- **Gene Therapy**: Recent developments in gene therapy for Parkinson's disease focus on targeted alteration of neuronal function. Clinical trials have used viral vectors for gene delivery directly into brain regions.
- **Experimental Drug BIIB094**: This drug targets Parkinson's gene and has shown promise in clinical trials, paving the way for gene-targeted therapies in neurodegenerative disorders.
- **Investigational Therapies**: Since the first gene therapy clinical trial in 2007, there has been escalating interest in manipulating cellular processes altered in Parkinson's disease.

## Summary
Recent research highlights several gene targets associated with Parkinson's disease, including LRRK2, SNCA, PINK1, PRKN, and ATP13A2. Advances in gene therapy and experimental drugs like BIIB094 offer promising avenues for treatment, focusing on targeted neuronal function alteration and gene-targeted therapies.