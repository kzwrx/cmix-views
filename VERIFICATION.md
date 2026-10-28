# Full-file verification

We completed an exact `enwik9` encode/decode round trip on 28 October 2026. This is a summary of our measured checks, not an independent benchmark certification.

## Files and checksums

| File | Bytes |
| --- | ---: |
| `comp9` | 3,946,605 |
| `archive9` | 90,515,901 |
| Reconstructed `enwik9_uncompressed` | 1,000,000,000 |

The complete `archive9` includes the compressed data, decoder, dictionary, model, and trailer. No separate decoder or model package is needed for decoding.

SHA-256:

```text
d9f7277fed3a945a8cce1f80b48c17c600ff5023d3ca70469485537476827955  comp9
7c4ab2658ebde542c862decd91ecf22ffddd334fae2c713dc168c2c64b534c25  archive9
159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc  enwik9_uncompressed
```

The reconstructed file and original `enwik9` have the same SHA-256 above, MD5 `e206c3450ac99950df65bf70ef61a12d`, and SHA-1 `2996e86fb978f93cca8f566cc56998923e7fe581`. A direct byte-for-byte comparison with `cmp` also returned exit status **0**.

## Measured run

Equivalent encode command with the input named `enwik9`: `./comp9 -e enwik9 main.cmix`. This also produces `archive9`. Decoding used `./archive9` with no arguments, as an ordinary user in a separate directory initially containing only that archive. The original input was kept outside the decode directory for comparison afterward.

| GNU time measurement | Encode | Decode |
| --- | ---: | ---: |
| Exit status | 0 | 0 |
| Elapsed time | 58 h 18 min 21 s | 54 h 50 min 38 s |
| Maximum resident memory | 14,685,672 KiB | 14,675,912 KiB |

Our reference system has an AMD Ryzen 9 9950X3D, an NVIDIA GeForce RTX 5090, and 64 GB of RAM, running Ubuntu 22.04 under WSL2. Both phases used one logical CPU core and no GPU. Memory figures are GNU time's maximum resident set size.

The frozen source and runtime files passed integrity checks. A separate clean build from the distributed source reproduced the measured `comp9` byte for byte.

## Checking the result

Follow the [build and verification instructions](README.md#running-and-verifying) to check the downloaded files and decode the archive. After decoding, check the output size and SHA-256 above, then compare it directly with the official `enwik9` using `cmp`.

