# Data cards

## DDXPlus
* Fansi Tchango, Goel, Wen, Martel, Ghosn. *DDXPlus: A New Dataset for
  Automatic Medical Diagnosis.* NeurIPS 2022 Datasets & Benchmarks
  (arXiv:2205.09148). Figshare article 22687585. **CC BY 4.0.**
* ~1.3 M synthetic patients (train 1,025,602 / validate / test 134,529),
  49 pathologies, 223 evidences. Generated from a proprietary knowledge base
  and a rule-based system; the released knowledge base has no conditional
  probabilities.
* In this repo (`artifacts/frozen/ddxplus/`): compact KB, token counts from
  the training split, reference-temperature fit, the 8 held-out pathologies,
  and 2,620 selected cases (pilot, panel, extension, distillation). Raw files
  are md5-checked by `ccrl data download ddxplus`.
* Known limits: synthetic; English translation of a French questionnaire;
  findings are positives only (absent findings are not rendered); public since
  2022 (training-data exposure possible — invariance estimands are unaffected
  because their target is zero; accuracy and direction agreement are exposed).

## MedCaseReasoning
* Wu, Wu, Thapa, Wei, Zhang, Suresh, Tao, Sun, Lozano, Zou. *MedCaseReasoning:
  Evaluating and learning diagnostic reasoning from clinical case reports*
  (arXiv:2505.11733). Hugging Face `zou-lab/MedCaseReasoning`, revision
  `469a5365bc534b5b2b9cfbc52b2ef2a10f43a339`. **MIT** per the dataset card.
* 14,489 cases from PubMed Central open-access case reports (train 13,092,
  val 500, test 897); case prompts written to stop before the diagnosis.
* In this repo (`artifacts/frozen/medcase/cases_selected.jsonl`): 2,376
  selected cases (pilot/panel/training preprocessing order) with PMCID, title,
  journal, date, link, case prompt and the authors' final diagnosis.
* Sending these texts to Amazon Bedrock is permitted: open access, published,
  de-identified case reports.
* Known limits: rare/complex diseases over-represented (case-report
  selection); single time-point presentations; LLM-assisted conversion
  (98 % hallucination-free in the authors' 100-case physician audit).

## Excluded: credentialed clinical data
MIMIC and any PhysioNet-credentialed data are **not** used in Track A: the
PhysioNet credentialed data use agreement prohibits sharing data with third
parties, including through APIs. `.gitignore` blocks such paths.

## Derived artifacts produced on the panel machine
`artifacts/frozen/banks/ddx_phrasebank.json` (+ review CSV),
`artifacts/frozen/medcase/prepared.jsonl` (+ review CSV),
`artifacts/frozen/items/*`, `artifacts/frozen/FREEZE.json`,
`config/models.lock.yaml`. All are deterministic functions of the committed
inputs and the archived preprocessing responses.
