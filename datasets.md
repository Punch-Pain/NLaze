# Datasets

NLaze's training data is **not published in this repository**. This file records
where the data comes from so the mix is reproducible and licensing is documented.

- Data files are stored locally under `data/external/`, pinned to upstream commits/tags
  and checksummed in `data/external/MANIFEST.sha256` (internal).
- The dataset build/validation scripts are internal; only the final training script
  ships with a release.
- Everything below was verified at source before use. No-license and
  non-commercial (NC) datasets were excluded.

## Sources

| Dataset | Upstream | License | Pin | Used for |
|---|---|---|---|---|
| Self-synthesized pairs | generated (typo/path/NL synthesis) | ours | — | typo, path, intent |
| nl2bash-custom | HF `AnishJoshi/nl2bash-custom` | MIT (upstream data re-licensed MIT 2020; GPL-3.0 code unused) | pinned | NL → command |
| NL2SH-ALFA | `westenfelder/NL2SH-ALFA` | MIT | pinned | NL → command |
| bash-commands-dataset | HF `aelhalili/bash-commands-dataset` | MIT | `67a539a` | NL → command |
| linux-command-dataset | HF `mecha-org/linux-command-dataset` | Apache-2.0 | `d3883ec` | NL → command |
| Mistake-To-Meaning | HF `ProCreations/Mistake-To-Meaning` | MIT | `7426669` | typo |
| spell-correction | HF `torinriley/spell-correction` | MIT | `4dd6bd9` | typo |
| bash_history | HF `spignelon/bash_history` | MIT | `4683648` | path tokens |
| shell-cmd-instruct | HF `byroneverson/shell-cmd-instruct` | Apache-2.0 | `459157f` | NL → command |
| text-to-command-gemini | HF `sakkke/text-to-command-gemini` | MIT | `b18777b` | NL → command |
| Linux_Terminal_Commands_Dataset | HF `darkknight25/Linux_Terminal_Commands_Dataset` | MIT | `db137e2` | NL → command |
| nl2shell-training-v3 | HF `AryaYT/nl2shell-training-v3` | Apache-2.0 | `d6fc438` | NL → command (dedup-filtered) |
| Command_Generation | HF `neerajnarwal/Command_Generation` | Apache-2.0 | `ee16ec8` | NL → command (Linux-filtered) |
| common-misspellings | Wikimedia Commons / Wikipedia lists | CC BY-SA 4.0 | 2026-09-22 | typo |
| tldr-pages | `tldr-pages/tldr` | CC BY 4.0 | `098d372` | path templates |
| typosquatting | `ecosyste-ms/typosquatting` | CC0 | `fd0bde9` | semantic fixes |
| thefuck rules | `nvbn/thefuck` | MIT | `c7e7e1d8` | flag/semantic fixes |

Evaluated and **not used**: kaggle-misspelled-words (conditional license,
upstream unlicensed), linlm-linux-commands (opaque provenance), nl-shell-multi
(attribution deferral), GitHub Typo Corpus (NC), plus every no-license dataset.

## Attribution

The data above is used locally for training and is **not redistributed**, so no
redistribution obligations are triggered. If derived data is ever shared
directly, carry: tldr-pages — CC BY 4.0; Wikipedia lists — CC BY-SA 4.0.

## Evaluation data (internal, pending review)

The per-type A/B report (v1 vs v2) and the external probe suite (145
hand-authored held-out items) with its raw results are kept internal under
`docs/`. Reviewed, identifier-pruned markdown summaries will be published in
`reports/`. The probe suite is deliberately excluded from training data.
