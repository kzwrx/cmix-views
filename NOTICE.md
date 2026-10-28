# cmix-views notices and attribution

cmix-views is a derivative lossless enwik9 compressor. It combines inherited cmix-family code with modifications, additions and integration described in [the modification notice](CHANGES.md). 

## License

The combined compressor and project-specific modifications use **GNU General Public License version 3 only** (`GPL-3.0-only`), not an automatic "or later" grant. The full, unaltered license is in [LICENSE](LICENSE).

Component-specific copyright, license, warranty and provenance notices remain applicable. Inherited components that carry other licenses retain those notices and terms; this project-level statement does not relicense them. The third-party license texts linked below cover emhash, LLVM/PyTorch components, mathfun and xxHash. The inherited zmix attribution retains its GPL-3.0-or-later wording.

## Upstream projects

These projects connect our implementation to the broader lineage credited below.

- **[cmix-neif-beehive](https://github.com/galqiwi/cmix-neif-beehive), Vladimir Malinovskii:** the published pretrained transformer, associated inference changes and selected context-mixing settings. cmix-views uses this model; it did not train a replacement base transformer.
- **[Ex-Girlfriend 1.0](https://github.com/rlanday/ex-girlfriend), Ryan Landay:** substantial inherited implementation changes, including the online fast-weight auxiliary predictor and its inference/mixer integration. These contributions remain attributed to Ryan Landay, not to cmix-views.
- **[cmix-neif, Marcus Wölk](https://github.com/marwoelk/cmix-neif-1.0), and the earlier cmix lineage, including [Byron Knoll](https://github.com/byronknoll/cmix):** much of the compressor, its models, preprocessing, arithmetic coding and archive machinery.
- **[zmix, James Byrne](https://zmix.frostbyrne.io/v1.0/README.md):** the inherited log-bilinear word-model design and hyperparameters, through cmix-neif's C++ port.

## Inherited component credits

The shared cmix-neif credit table below is reproduced once from [Beehive's notice](https://github.com/galqiwi/cmix-neif-beehive/blob/main/NOTICE.md); [Ex-Girlfriend's notice](https://github.com/rlanday/ex-girlfriend/blob/main/NOTICE.md) preserves the same table. Its references to "this entry", "here" and source paths describe the upstream projects, not new work by cmix-views. A dash in the license column is part of the upstream attribution, not a new license grant.

The upstream notice states:

> This program is a derivative work and is distributed under the GNU General
> Public License (see `LICENSE`). The copyright and licence notices in the
> individual source files are unchanged and remain in force.

| component | author | licence |
|---|---|---|
| `cmix`: coder, contexts, states, mixer, byte/direct/indirect/match models | Byron Knoll | GPL-3 |
| `cmix-hp`: changes to cmix on top of STARLIT | Byron Knoll | GPL-3 |
| `fast-cmix`: a faster implementation of `cmix-hp`, from which `fx-cmix` was developed; it brought the hash maps and the containers of `src/ds/` | Saurabh Kumar | GPL-3 |
| `fxcm` / `fxcm_v26`: text model, in `cmix-lex` integrated into the cmix predictor (`src/models/fxcmv1.cpp`) | Kaido Orav; the integration Ibrahim Marcouch | GPL-2-or-later |
| `fx-cmix` | Kaido Orav | GPL-3 |
| `fx2-cmix` | Kaido Orav, Byron Knoll | GPL-3 |
| `cmix-lex`: the immediate fork base, derived from `fx2-cmix` (Kaido Orav, Byron Knoll), with Kaido Orav's text model `fxcm_v26`; Hutter Prize 2026, awarded to Ibrahim Marcouch and Kaido Orav | Ibrahim Marcouch | GPL-3 |
| **`fx2-cmix-transformer`: the transformer inference engine** (`github.com/astOwOlfo/fx2-cmix-transformer-v1`; 25 files under `src/tf1/`, each with a provenance banner, and, in the coder-side driver written after his `src/predictor.cpp`, his float16 helpers, error helper, float16 file writer, loss report and the per-byte step around the article separator; beside the driver, 10 further files of that directory are this entry's own), including the layer pattern of linear-attention and windowed attention layers and the container formats v1 and v2; and the training code with which the weights were trained, shipped verbatim in `training/tf/tree_upstream/`, from which the training form in `training/tf/tree_overlay/` and the export tools are derived | Vladimir Ivanov | GPL-3 |
| the single-precision sine and cosine of NVIDIA's CUDA libdevice (`__nv_sinf`, `__nv_cosf`), ported bit-exactly by Vladimir Ivanov in his container code to recompute the rotary tables (`src/tf1/weights_io_compressed.cpp`, `training/tf/4_export/weights_compress_v3.py`) | NVIDIA, transcribed by Vladimir Ivanov | libdevice: CUDA Toolkit End User License Agreement, no licence of its own; the transcription contains no NVIDIA source text and was published by Ivanov under GPL-3.0; see `training/ATTRIBUTION.md`, section 7 |
| **the base article order** that the order of this entry modifies: the article order of `fx2-cmix` (title and first 4,000 characters of each article, embedded, arranged on one axis and clustered; a manual sort step moves census, image and disambiguation pages to the end; `fx2-cmix` commits "article order v5" to "v7", July 2024), inherited through `cmix-lex` in the revision of the `fxcm_v26` archive; the order rule of this entry recognises those page classes with the strings of that manual sort step; in that revision Kaido Orav gathered the `Wikipedia:` pages in one block at the end | Byron Knoll, Kaido Orav | GPL-3 |
| the 510 fine clusters of the inherited dictionary order | Kaido Orav | GPL-3 |
| STARLIT: the idea of reordering articles, and the inherited code in `src/readalike_prepr/` that reorders the articles, sorts them back by page id, splits the input and embeds dictionary and article order in the executable | Artemiy Margaritov | GPL-3 |
| phda9: Wikipedia preprocessing and the metadata block | Alexander Rhatushnyak | GPL-3 as distributed in STARLIT (the transforms carry no licence notice of their own) |
| paq8l / paq8hp12any: WRT, state tables | Matt Mahoney, Alexander Rhatushnyak, Przemysław Skibiński and others | GPL |
| paq8px: in the text model (`src/models/fxcmv1.cpp`) the English stemmer, after the Porter2 algorithm of Martin Porter, and the second match model, based on paq8px v208 | Márcio Pais (stemmer); Márcio Pais, Zoltán Gotthardt (match model) | GPL |
| PPMd: model and allocator | Dmitry Shkarin | public domain / GPL as distributed |
| `mod_ppmd` and the SSE stage | Eugene Shelwien | GPL |
| from `fx2-cmix` (`github.com/rohitdwivedula/fx2-cmix`, commit `468e92b9`): an SSE2 search for check values in the context maps (its compare in the form of that commit, extended to 16 slots) and the reuse of the last context lookup in the mixer (in the structure of that commit, with names and a validity flag of this entry), both **rewritten** here; the header inlining of `Sigmoid::Logit` (with a body of this entry) and of `MixerInput::Set*` (**taken over** in the form of the commit) | Rohit Dwivedula | GPL-3 |
| **grammar-match model**: line-for-line port from `fx-deepmix` | Halvor Yttredal | GPL-3 |
| the feed of the byte-model prediction into the internal mixers of the text model (two lines in the form of `fx-deepmix`; the wiring is Kaido Orav's from `fx2-cmix`), and the value 0.1 of the last step of the mixer learning-rate schedule | Halvor Yttredal | GPL-3 |
| **BITLSTM, the LSTM correction model behind the mixer** (published in `cmix-obias` as `bitlstm32`): its name, its form, its AVX2 kernels, several of its input features and its offline training on a trace with trained weights shipped | David Freelan | GPL-3 |
| **the PPMd prior on the output logits of the byte LSTM** (`src/ppm_logit_prior.h` and the output-bias hook in `src/mixer/byte-mixer.*` and `src/mixer/lstm.*`): taken from `cmix-obias` (`src/models/obias-prior.*` and the same hook there) in the form of its constant-weight build, here with the fixed weight 0.55 | David Freelan | GPL-3 |
| the branch-free evaluation of the context-map states: the zero-state masking taken over in the form of `cmix-obias`, the run-length masking after it; the presence-mask duplicate check is this entry's own | David Freelan | GPL-3 |
| **from `zmix 1.0`**: adaptive depth (a logistic model in front of the mixer cascade that skips the cascade on confident bits), indel recovery in the match model, the end-to-end gradient of the mixer cascade, the word-label model with its hyper-parameters (after Mnih and Hinton 2007 and Morin and Bengio 2005, as cited there), the time constant of one context-map family, the width of one flag field of the secondary estimation and the option of narrowing a second one, and the first mixing stage of a check-value hash; rewritten in C++ for this entry (the word-label model and the backward recursion of the gradient function by function, the indel recovery line by line, from the Zig source), each item with a banner naming its origin in the file that defines it | James Byrne | GPL-3-or-later |
| narrower index types of several hash tables and a larger shared map: ideas from Layth Lakhani's experiments on `fx3-cmix` (the program is Kaido Orav's), implemented independently | Layth Lakhani | GPL-3; his changes GPL-3.0-or-later (only ideas are used) |
| the question of the widths of the direct state maps of the text model: from the public `fx3-cmix` experiments (`github.com/Seth-Lupo/fx3-cmix`), where they were made smaller; the larger widths were measured for this entry | Seth Lupo | GPL-3 (only the idea is used) |
| the stretch curve ln(p^k / (1 - p^z)), here with k = 0.75 and mirrored at p = 0.5, in the stretch table of the context maps (`mpais_sym` in `src/models/fxcmv1.cpp`), after a public encode.su post (thread 4534, "Research ideas", post 88828, 26 August 2026), reimplemented and measured | mpais | - |
| the block-diagonal recurrence of the small byte LSTM: the form published in `github.com/Xeloraa/Hutter-compression-research`, implemented independently and measured | Xeloraa | - |
| the type of the linear-attention layers of the transformer, Kimi Delta Attention from Kimi Linear (arXiv 2510.26692), used through the design of `fx2-cmix-transformer`, whose C++ form reproduces the semantics of the kernels of `flash-linear-attention` | Kimi Team; Songlin Yang, Yu Zhang, Zhiyuan Li (`flash-linear-attention`) | -; `flash-linear-attention`: MIT |
| `emhash` hash map / hash set | Huang Yuanbing (`ktprime`) | MIT, full text in `third_party/` |
| `SmallVector` (via PyTorch `c10`, version 2.0.0, through `fast-cmix`) | LLVM Project / Meta | Apache-2.0 WITH LLVM-exception; the PyTorch changes BSD-3-Clause; full texts in `third_party/` |
| `AlignOf` (via PyTorch `c10`, version 2.0.0, through `fast-cmix`) | LLVM Project (University of Illinois) / Meta | NCSA; the PyTorch changes BSD-3-Clause; full texts in `third_party/` |
| the form of the adaptive binary range coder of the transformer weight container (11-bit probabilities, shift 5), which Ivanov's container code follows | Igor Pavlov (LZMA SDK) | public domain |
| squash function with its 33-point table, inverse stretch table, binary arithmetic coder and APM form of the self-extractor's decoder (`paq7`) | Matt Mahoney | GPL |
| the idea of the x86 BCJ filter of the LZMA SDK (relative call and jump targets made absolute before coding), which the exact address transform of the self-extractor's stub extends (`tools_src/packer/x86t.h`) | Igor Pavlov | public domain (only the idea is used) |
| a context-mixing decoder in a small executable stub, the principle of Crinkler, followed by the self-extractor's stub (`tools_src/packer/`) | Rune L. H. Stubbe, Aske Simon Christensen | - (only the idea is used) |
| civil-date conversion of the timestamps (`days_from_civil`, `civil_from_days` in `src/timestamp_algebra.h`) | Howard Hinnant | public domain |
| division by run-time constants through a magic multiplier (`src/fast_div.h`): the method of Granlund and Montgomery (1994) in the 64-bit form of `magicu` from *Hacker's Delight*, section 10-9 | T. Granlund, P. L. Montgomery; Henry S. Warren, Jr. | - (a method; the code of the book's website may be used freely) |
| one xor-shift and multiply step with the multiplier `0x85ebca6b` of the MurmurHash3 finalizer, in the check-value hash of the compact context maps (`src/context_map_compact.h`) | Austin Appleby | public domain |
| the primes and the first three avalanche steps of xxHash32 in the context hash of the packer (`cm_hashn` in `tools_src/packer/cm_core.h`), and its primes with the first avalanche steps in the pair and triple hashes of the small byte LSTM (`BigCol`, `TriCol` in `src/lstm_aux_contexts.h`); in helpers that the program does not call, its full avalanche step (`Mix32` in `src/invariants.h`) and two primes (`UniByte`, `UniProb` in `src/lstm_fx_link.h`) | Yann Collet | BSD-2-Clause, text in `third_party/` (xxHash v0.8.4) |
| the split of ln 2 into `kLn2Hi` and `kLn2Lo`, and `kLog2e` and `kLn2`, in `src/detmath.cpp`: the values and decimal spelling of `ln2HI`, `ln2LO` and `invln2` of `e_exp.c` and `lg2` of `e_pow.c`; no fdlibm code is used, the functions are series of this entry | Sun Microsystems (fdlibm) | - (numerical constants) |
| polynomial forms of `expf`, `logf` and `tanhf`: the vector `exp` and `log` of the transformer engine (`src/tf1/opt/vec_math.h`, from `fx2-cmix-transformer`), `Exp8` of BITLSTM (`src/bit_head.cpp`, from `cmix-obias`) and the eight-wide `tanh` of the byte LSTM (`src/fast_math_vec.cpp`) | Stephen L. Moshier (Cephes Mathematical Library) | free use, as stated by the Cephes distribution |
| altered versions of `exp256_ps` and `log256_ps` of `avx_mathfun.h` (based on `sse_mathfun.h`): `Exp8` of BITLSTM (`src/bit_head.cpp`, through `cmix-obias`), `exp256_ps`, `log256_ps` and `exp_full8` of the transformer engine (`src/tf1/opt/vec_math.h`, `src/tf1/opt/kda_math.h`, through `fx2-cmix-transformer`) | Giovanni Garberoglio; Julien Pommier | zlib, notices in `third_party/` |
| the FNV-1a hash scheme with the FNV primes, in `src/alias_table.h`, `src/string_pool.h`, `src/metablock.h`, `src/readalike_prepr/order_codec.h`, `src/readalike_prepr/phda9_preprocess.h`, `src/preprocess/dict_derive.cpp`, `src/preprocess/dictionary.h`, `src/ppmd_prewarm.h`, `src/tf1/tf1_engine.cpp`, `src/tf1/tf1_z3.cpp` and `training/tf/tree_overlay/zctx/z3_extract.py` | Glenn Fowler, Landon Curt Noll, Kiem-Phong Vo | public domain |
| splitmix64, for the fixed random projections and the initial weights of BITLSTM (`src/bit_head.cpp`, `training/bitl/trainer/bit_head.cpp`); two of its constants also in the hash of the word-label model (`src/word_label.cpp`, through `zmix`) | Sebastiano Vigna (after G. L. Steele, D. Lea, C. H. Flood) | public domain |
| the Muon optimizer, used through `torch.optim.Muon` for the weight matrices in the training of the transformer | Keller Jordan and co-authors | - (a method) |

## Third-party license texts

These license files are retained unaltered. Existing copyright, license, warranty and modification notices in individual source files must also remain intact.

| Component | License text |
| --- | --- |
| emhash | [MIT](third_party/emhash-LICENSE.txt) |
| LLVM SmallVector and legacy AlignOf, through PyTorch c10 | [Apache-2.0 with LLVM exception and legacy NCSA](third_party/LLVM-LICENSE.txt), plus [PyTorch BSD-3-Clause](third_party/PyTorch-LICENSE.txt) |
| avx_mathfun / sse_mathfun | [zlib notices](third_party/mathfun-LICENSE.txt) |
| xxHash | [BSD-2-Clause](third_party/xxHash-LICENSE.txt) |

## Upstream modifications and training credits

- **Marcus Wölk (Marcus Woelk), 2026:** cmix-neif changes to enwik9 preprocessing, dictionary derivation, models and mixers, PPMd and memory handling, the pretrained transformer and BITLSTM, the correction model, archive packaging, and runtime and code-size improvements. His authorship covers those modifications, not inherited code.
- **Vladimir Malinovskii, October 2026:** Beehive's replacement pretrained transformer and associated inference kernels and loader, three compiled-in context-mixing settings, dense-weight prefetch tuning, and model-training changes derived from cmix-neif and Vladimir Ivanov's training code.
- **Ryan Landay, Ex-Girlfriend 1.0:** the inherited fast-weight auxiliary predictor and its inference/mixer integration, metadata and timestamp prediction changes, table-capacity changes and build changes. The original dated modification notice states:

> Modified by Ryan Landay, developed with Claude (Anthropic) and Codex (OpenAI).
> Project techniques: September 24–October 10, 2026. Neif port: October 6–10.
> Release documentation and verification receipt: October 14, 2026.
> Our modifications use GPL-3.0-only; all inherited notices remain in force.

cmix-neif's transformer training and export tools derive from **Vladimir Ivanov's fx2-cmix-transformer**; its BITLSTM trainer uses the model form and AVX2 kernels credited to **David Freelan**. Upstream training also credits **Keller Jordan and co-authors** for modded-nanogpt and Muon, the **Kimi Team** and the **flash-linear-attention** authors for Kimi Linear/KDA, and the published work on rotary position embeddings, query/key normalization and ParetoQ. These training-method credits are distinct from copied-code licenses.

cmix-views reuses Beehive's pretrained transformer unchanged. Further historical details remain in the upstream [Beehive training documentation](https://github.com/galqiwi/cmix-neif-beehive/blob/main/training/README.md), [training attribution](https://github.com/galqiwi/cmix-neif-beehive/blob/main/training/ATTRIBUTION.md) and [Ex-Girlfriend modification notice](https://github.com/rlanday/ex-girlfriend/blob/main/EX-GIRLFRIEND.md). Our own dated changes remain in [CHANGES.md](CHANGES.md).

## Data and unresolved provenance

Wikipedia text is not relicensed under the compressor's GPL statement. The inherited notices attribute it to Wikipedia contributors and describe GFDL / CC BY-SA terms. Any future distribution of the corpus, text excerpts or an archive carrying that text requires its own applicable data-license and attribution review.

This project inherits code and mechanisms from earlier cmix-family work and carries forward the upstream attribution notices and unresolved provenance and licensing questions documented in [the inherited training attribution](https://github.com/galqiwi/cmix-neif-beehive/blob/main/training/ATTRIBUTION.md#7-open-points).

### Inherited data attribution

The following data notice is retained from cmix-neif through Beehive. Paths and descriptions of files shipped refer to that upstream source distribution.

`prof_input/` holds two short excerpts of the English Wikipedia, inherited with
the fork base (they are taken from `enwik8`, which is itself the first 100 MB of
`enwik9`); `training/tf/tree_upstream/prof_input/` is an identical copy.  The
build reads `input` only to train a compiler profile when `pgo_data/` holds
none; with the shipped profile neither file is read, and not one of their
bytes ends up in the compressed archive.  Wikipedia text is published under
the **GNU Free Documentation License** and the **Creative Commons
Attribution-ShareAlike** licence; the authors are the contributors to the
articles concerned, listed in the page histories of `en.wikipedia.org`.  The
same is true for the two excerpts `enwik8`/`enwik9` themselves, which the
Hutter Prize distributes.

`dictionary/english.dic`, `dictionary/english.hints` and the article-order
residual were derived from `enwik9` and from the inherited word list and article
order (`training/ATTRIBUTION.md` section 5); they contain word forms and
numbers, not text.  Bands 3 and 4 of the dictionary were ordered offline with
the pre-trained embedding models `intfloat/e5-large-v2` and
`BAAI/bge-reranker-base` (MIT), of which nothing ships.

## Development assistance

OpenAI Codex assisted with the development, testing and documentation of cmix-views. This acknowledgment is additional to upstream attribution. 

For questions, bug reports, attribution corrections, or other inquiries, please contact us at smf3567@gmail.com. 
