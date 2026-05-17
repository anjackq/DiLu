# Related Work Candidates

This file started as a candidate shortlist. The core references below have now been **verified from sourceable records** and are safe to use as manuscript anchors. Any item not listed in the verified set should still be treated as a candidate only.

## Rules

- Do not copy any item below directly into the paper bibliography without verification.
- Use trusted programmatic sources first.
- If a candidate cannot be verified, keep it as a placeholder and exclude it from the final reference list.

## Verified Core Set

These five references are sufficient to support the current draft's baseline framing.

| Key | Reference | Why it matters | Verified source |
| --- | --- | --- | --- |
| `wen2023dilu` | Wen, L., Fu, D., Li, X., Cai, X., Ma, T., Cai, P., Dou, M., Shi, B., He, L., and Qiao, Y. *DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models*. arXiv:2309.16292. Zotero key: `TDPT8STR`. | Direct parent framework and closest upstream driving-agent baseline. | Zotero metadata for `TDPT8STR`; arXiv record for `2309.16292` |
| `leurent2018highwayenv` | Leurent, E. *An Environment for Autonomous Driving Decision-Making*. GitHub / `highway-env`, 2018. | Canonical citation for the simulation environment used in the paper. | official `highway-env` documentation citation block |
| `dettmers2023qlora` | Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314. | Efficient fine-tuning background most relevant to local model adaptation. | DOI content negotiation for `10.48550/arXiv.2305.14314` |
| `song2023powerinfer` | Song, Y., Mi, Z., Xie, H., and Chen, H. *PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU*. arXiv:2312.12456. | Strong anchor for local or constrained LLM runtime discussion on consumer hardware. | DOI content negotiation for `10.48550/arXiv.2312.12456` |
| `sha2023languagempc` | Sha, H., Mu, Y., Jiang, Y., Chen, L., Xu, C., Luo, P., Li, S. E., Tomizuka, M., Zhan, W., and Ding, M. *LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving*. arXiv:2310.03026. | Closely related LLM-for-driving decision-making paper. | arXiv record for `2310.03026` |

## Verified Extension Set

These references were verified to support a compact but credible Related Work section.

| Key | Reference | Why it matters | Verified source |
| --- | --- | --- | --- |
| `hu2021lora` | Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685. | Canonical PEFT background reference beneath QLoRA. | DOI content negotiation for `10.48550/arXiv.2106.09685` |
| `fu2023drivelikehuman` | Fu, D., Li, X., Wen, L., Dou, M., Cai, P., Shi, B., and Qiao, Y. *Drive Like a Human: Rethinking Autonomous Driving with Large Language Models*. arXiv:2307.07162. | Pre-DiLu autonomous-driving LLM paper from the same research line; useful for historical positioning. | DOI content negotiation for `10.48550/arXiv.2307.07162` |
| `xu2023drivegpt4` | Xu, Z., Zhang, Y., Xie, E., Zhao, Z., Guo, Y., Wong, K.-Y. K., Li, Z., and Zhao, H. *DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model*. arXiv:2310.01412. | Additional autonomous-driving LLM reference with a different end-to-end framing. | DOI content negotiation for `10.48550/arXiv.2310.01412` |
| `yang2023llm4drive` | Yang, Z., Jia, X., Li, H., and Yan, J. *LLM4Drive: A Survey of Large Language Models for Autonomous Driving*. arXiv:2311.01043. | Compact survey anchor for positioning the field without over-expanding the related-work section. | DOI content negotiation for `10.48550/arXiv.2311.01043` |

## Verified Usage Guidance

- Use `wen2023dilu` when describing the upstream DiLu framework and motivation for language-model driving agents.
- Use `leurent2018highwayenv` when introducing the simulation environment.
- Use `dettmers2023qlora` in the fine-tuning background section.
- Use `song2023powerinfer` when motivating local-first or consumer-grade inference constraints.
- Use `sha2023languagempc` as a closely related autonomous-driving LLM baseline in the related-work section.
- Use `hu2021lora` when the paper needs the canonical PEFT background under QLoRA.
- Use `fu2023drivelikehuman`, `xu2023drivegpt4`, and `yang2023llm4drive` to keep Related Work grounded in actual autonomous-driving LLM literature instead of broad generic LLM planning references.

## 1. LLM Agents for Driving / Autonomous Decision Making

### Candidate anchors

| Candidate | Why it matters | Current status | Verification note |
| --- | --- | --- | --- |
| DiLu original paper | Direct parent framework and closest prior system baseline | repo-local citation exists in `README.md` | Verify title, authors, venue/arXiv metadata before manuscript use |
| highway-env environment paper | Needed if the paper cites the simulation benchmark environment itself | pending verification | Search for the canonical highway-env citation and benchmark description |
| LLM planning / autonomous-driving agent papers | Needed for positioning DiLu-Ollama among language-model driving systems | pending verification | Build a small shortlist after targeted search rather than citing broadly |

### Search prompts

- `language model autonomous driving simulation paper`
- `llm driving agent highway-env`
- `autonomous driving large language model planning`

## 2. Local or Resource-Constrained LLM Evaluation

### Candidate anchors

| Candidate | Why it matters | Current status | Verification note |
| --- | --- | --- | --- |
| local LLM deployment / inference studies | Supports the local-first framing and runtime constraints discussion | pending verification | Prefer papers that explicitly discuss latency, on-device or local serving, or constrained inference |
| model serving / runtime reliability studies | Useful for motivating why invalid runs and timeout behavior matter scientifically | pending verification | Use only if they directly connect to inference latency or runtime stability |

### Search prompts

- `local llm inference latency evaluation paper`
- `resource constrained language model evaluation`
- `llm runtime latency benchmark local deployment`

## 3. Fine-Tuning / PEFT for Task Specialization

### Candidate anchors

| Candidate | Why it matters | Current status | Verification note |
| --- | --- | --- | --- |
| LoRA | Canonical PEFT method background | pending verification | Verify only if cited for method background rather than implementation detail |
| QLoRA | Relevant for efficient adaptation of local models | pending verification | Verify if the fine-tuning section needs memory-efficient adaptation context |
| PEFT survey / empirical adaptation papers | Useful for positioning fine-tuning expectations | pending verification | Prefer concise, high-signal papers over broad surveys |

### Search prompts

- `LoRA paper language model adaptation`
- `QLoRA paper quantized low rank adaptation`
- `parameter efficient fine tuning local language models`

## 4. Simulation Benchmark Methodology

### Candidate anchors

| Candidate | Why it matters | Current status | Verification note |
| --- | --- | --- | --- |
| benchmark validity / runtime-aware evaluation papers | Helpful for framing invalid runs as part of the measurement problem | pending verification | Prefer work that explicitly discusses evaluation validity, not generic benchmarking |
| simulation-based autonomous-agent benchmark papers | Useful for framing the experimental design and screening protocol | pending verification | Keep only papers that are methodologically close to the paper’s scope |

### Search prompts

- `simulation benchmark validity machine learning runtime`
- `autonomous agent benchmark invalid runs latency evaluation`
- `benchmarking autonomous driving agents simulation methodology`

## Immediate next verification queue

The paper now has enough verified literature to support a short conference-style Related Work section. The next useful additions, if we want to deepen the final manuscript later, are:

1. one runtime-aware evaluation or benchmark-validity paper closer to empirical methodology
2. one additional local inference systems paper beyond `song2023powerinfer`
3. one non-autonomous-driving agent benchmark paper for broader evaluation framing

## Draft placeholders

Use placeholders like these in prose until verification is complete:

- `[CITATION NEEDED: original DiLu paper]`
- `[CITATION NEEDED: highway-env paper]`
- `[CITATION NEEDED: PEFT background]`
- `[CITATION NEEDED: local LLM runtime evaluation]`
