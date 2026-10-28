# cmix-views 1.0

cmix-views is an incremental refinement of [cmix-neif-beehive](https://github.com/galqiwi/cmix-neif-beehive) for the [Large Text Compression Benchmark (LTCB)](https://mattmahoney.net/dc/text.html), maintained by Matt Mahoney. It compresses `enwik9`: the first 1,000,000,000 bytes of the English Wikipedia XML dump dated March 3, 2006.

The implementation combines existing cmix-family work and changes from [Ex-Girlfriend 1.0](https://github.com/rlanday/ex-girlfriend) by Ryan Landay with our additional online prediction, calibration, and recurrent word-history learning.

## Results

Full-file round-trip verification completed on 28 October 2026. The reconstructed file matches the original byte for byte, with matching SHA-256, MD5, and SHA-1 hashes.

| Measurement | Result |
| --- | --- |
| Original input | 1,000,000,000 bytes |
| Complete self-extracting archive | 90,515,901 bytes |
| [cmix-neif-beehive](https://github.com/galqiwi/cmix-neif-beehive) archive (parent baseline) | 90,763,792 bytes |
| [cmix-wolf P14](https://github.com/m-philipp-gd/cmix-wolf-2/releases/tag/p14) archive | 90,735,088 bytes |
| Reduction relative to Beehive | 247,891 bytes (0.2731%) |
| Reduction relative to P14 | 219,187 bytes (0.2416%) |
| Encode elapsed time, GNU time | 58 h 18 min 21 s |
| Decode elapsed time, GNU time | 54 h 50 min 38 s |
| Encode peak resident memory, GNU time | 14.005 GiB |
| Decode peak resident memory, GNU time | 13.996 GiB |
| Exact full-file reconstruction | Verified: all 1,000,000,000 bytes match |
| enwik8 result | Not measured |

The complete archive includes the compressed data, decoder, dictionary, model, and trailer, all counted once.

## Changes from cmix-neif-beehive

cmix-views retains [Beehive's](https://github.com/galqiwi/cmix-neif-beehive) pretrained transformer and substantial compressor infrastructure. From Ryan Landay's [Ex-Girlfriend 1.0](https://github.com/rlanday/ex-girlfriend), we incorporate the online fast-weight auxiliary predictor and its inference/mixer integration, enlarged context-model tables, and revision-ID and timestamp prediction changes.

Our additional mechanisms include:

- **Recurrent word history:** a compact, 32-dimensional recurrent state built from inherited word features. Online learning propagates through up to 16 recorded word transitions. 
- **Additional prediction views:** alternative predictions derived from existing transformer features, with bounded adaptive calibration.
- **Word-context correction:** context-dependent residual corrections to coding predictions, using information from the inherited word model.
- **Prediction interactions:** cubic interactions between disagreements among prediction views, with guarded numerical caching to reduce repeated computation.

These mechanisms are integrated with Beehive's model dimensions and coding path. Its efficient pretrained transformer remains unchanged. 

## Hardware and measurement

Our reference system has an AMD Ryzen 9 9950X3D, an NVIDIA GeForce RTX 5090, and 64 GB of RAM, running Ubuntu 22.04 under WSL2.

The measured encoding and decoding runs used one logical CPU core. We used the GPU for research and exploration, but the codec does not use it. Elapsed time and peak resident memory are measured with GNU time.

## Build from source

The source, pretrained model, and construction inputs are included. The reference build uses Linux x86-64, Python 3.8 or later, Clang/LLVM/LLD 18.1.8, GNU make, GCC, GNU binutils, GNU awk, Bash, and standard coreutils. The versioned LLVM commands must be available, and `awk` must support GNU `strtonum`. Install these prerequisites separately; the build does not download software or require administrator access.

From the repository root on a Linux filesystem, run:

```sh
python3 build.py
```

Use a path without spaces. The command creates a new `build-work/` directory and writes `build-work/comp9`. It refuses to overwrite an existing build directory. To use another new directory, pass `--work-dir /path/to/new/build-directory`.

The script preserves the measured compiler settings, checks the packed decoder, verifies both embedded construction inputs by decoding them, and checks the final compressor against the SHA-256 below. This standalone build reproduced the measured `comp9` byte for byte. 

## Running and verifying

The measured `comp9` and `archive9` binaries are attached under [Releases](https://github.com/kzwrx/cmix-views/releases). You can also build `comp9` from source as described above.

The tested executables are for Linux x86-64 with x86-64-v3 CPU support, including AVX2 and FMA. 

Please allow additional RAM beyond the measured process peak for the operating system, and free disk space for temporary files and the reconstructed output.

To encode, use a fresh working directory containing the matching `comp9` and the official `enwik9`:

```sh
chmod u+x comp9
./comp9 -e enwik9 main.cmix
```

The command creates `main.cmix` and assembles the complete self-extracting `archive9`. The archive size, not the intermediate stream size, is the reported result.

To decode, put `archive9` in a separate, otherwise empty writable directory. Keep the original input elsewhere. Run as an ordinary user:

```sh
chmod u+x archive9
./archive9
```

Decoding requires no arguments or network access and writes `enwik9_uncompressed`. After a successful exit, verify it against your original:

```sh
wc -c < enwik9_uncompressed
sha256sum enwik9_uncompressed
cmp enwik9_uncompressed /path/to/original/enwik9
```

Replace the example reference path with the location of your original file. The output must contain 1,000,000,000 bytes; `cmp` must return exit status 0.

SHA-256 checksums of the measured artifacts:

```text
d9f7277fed3a945a8cce1f80b48c17c600ff5023d3ca70469485537476827955  comp9
7c4ab2658ebde542c862decd91ecf22ffddd334fae2c713dc168c2c64b534c25  archive9
159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc  enwik9_uncompressed
```

## Acknowledgments

cmix-views builds on substantial upstream work. Upstream contributions include:

- **[cmix-neif-beehive](https://github.com/galqiwi/cmix-neif-beehive), by Vladimir Malinovskii:** the pretrained transformer and selected context-mixing changes.
- **[Ex-Girlfriend 1.0](https://github.com/rlanday/ex-girlfriend), by Ryan Landay:** the online fast-weight auxiliary predictor and its inference/mixer integration, enlarged context-model tables, and revision-ID and timestamp prediction changes.
- **[Marcus Wölk's cmix-neif](https://github.com/marwoelk/cmix-neif-1.0) and the earlier cmix lineage, including [Byron Knoll's work](https://github.com/byronknoll/cmix):** much of the underlying compressor.
- **[James Byrne's zmix](https://zmix.frostbyrne.io/v1.0/README.md):** the word-model design and hyperparameters inherited through cmix-neif's C++ port.

These acknowledgments are not a complete contributor list; we also thank the many earlier contributors whose work these projects build on. Detailed credits, retained upstream notices, and previously documented provenance questions are recorded in [NOTICE.md](NOTICE.md). 

For questions, bug reports, or attribution corrections, please contact us at [smf3567@gmail.com](mailto:smf3567@gmail.com). 

## AI use and development approach

For this project, we used OpenAI's Codex to accelerate coding, analysis, and experimentation. Most of our work on this entry used GPT 6.1 Sol, selected for cost efficiency.

Our project also draws on earlier research conducted with models and tools from multiple providers such as Anthropic. Useful ideas, unsuccessful experiments, and positive and negative results all informed our choices. These tools accelerated our development cycle, and we acknowledge the researchers and engineers behind them.

Although execution speed does not affect [LTCB ranking](https://mattmahoney.net/dc/text.html), encoding and decoding times influenced which approaches we pursued because they limited how quickly we could test ideas. We believe there are opportunities for further optimization.

We reused Beehive's pretrained transformer and concentrated on online prediction, calibration, and recurrent word-history mechanisms around that foundation. This approach allowed us to build on substantial upstream work within our available time, experience, and computing budget.

Finally, we regard cmix-views as an incremental refinement built on substantial upstream work. We stand on the shoulders of giants. 
