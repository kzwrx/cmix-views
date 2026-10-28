# cmix-views modification notice

Production changes incorporated by: **23 October 2026**

## Bases and inheritance

The production integration uses [cmix-neif-beehive](https://github.com/galqiwi/cmix-neif-beehive), by Vladimir Malinovskii, together with inherited changes from [Ex-Girlfriend 1.0](https://github.com/rlanday/ex-girlfriend), by Ryan Landay. These include Ex-Girlfriend's online fast-weight auxiliary predictor and its inference/mixer integration, enlarged context-model tables, and revision-ID and timestamp prediction changes. They are not new contributions of cmix-views. The configuration also restores two older double-indirect context models from the cmix lineage.

Both projects descend from Marcus Wölk's cmix-neif and its earlier cmix-family dependencies.

## cmix-views additions and integration

The changes include:

- Additional prediction views and bounded adaptive calibration, reusing inherited transformer computation.
- Context-dependent word-residual correction and cubic interactions between prediction-view disagreements.
- Guarded numerical caching for the selected cubic computation.
- Stable-reference prediction access and supporting interfaces. LSTM-reference interface code is present but was inactive in the measured run. Further work could explore whether enabling and refining this path reduces the complete archive size.
- A 32-coordinate recurrent word-history model using the inherited 96-dimensional word features, with online learning through at most 16 recorded transitions.
- Integration with Beehive's model geometry, including four 1,024-coordinate activation banks for the inherited fast-weight path; the published base transformer remains unchanged.
- Coding-path integration, deterministic update handling and diagnostic/progress reporting needed for the combined implementation.

The measured archive-size reduction applies to the combined implementation. 

## License and retained notices

Release packaging was updated on 28 October 2026 with a standalone build script and reproducible construction inputs. The build reproduces the measured compressor byte for byte; these packaging changes do not change its compression behavior.

cmix-views-specific modifications use GPL-3.0-only. All applicable upstream notices remain in force. This dated project-level notice supplements, rather than replaces, the preserved modification notices and source-file attribution.
