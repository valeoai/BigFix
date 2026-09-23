# Third-party notices

BIGFix itself is released under the MIT license (see [LICENSE.txt](LICENSE.txt)). This repository also contains
code and data that were adapted from, or come from, other projects. They remain under their original licenses,
which are reproduced or linked below.

| Material | Where in this repository | Origin | License |
|---|---|---|---|
| VQGAN tokenizer | `bigfix/network/vq_model.py` | [LlamaGen](https://github.com/FoundationVision/LlamaGen) | MIT |
| Top-k / top-p (nucleus) logits filtering | `bigfix/sampler/halton_sampler.py` (`top_k_top_p_filtering`) | LlamaGen `autoregressive/models/generate.py` | MIT |
| Parts of the VQGAN architecture (through LlamaGen) | `bigfix/network/vq_model.py` | [taming-transformers](https://github.com/CompVis/taming-transformers) | MIT |
| Parts of the transformer | `bigfix/network/transformer.py` | [nanoGPT](https://github.com/karpathy/nanoGPT) | MIT |
| Adaptive modulation layers | `bigfix/network/transformer_block.py`, `bigfix/network/txt_transformer.py` | [DiT](https://github.com/facebookresearch/DiT) | CC BY-NC 4.0 |
| Precision / recall / density / coverage | `bigfix/metrics/inception_metrics.py` | [prdc](https://github.com/clovaai/generative-evaluation-prdc) (NAVER) | MIT |
| Matrix square root, FID and Inception Score computation | `bigfix/metrics/inception_metrics.py` | [torchmetrics](https://github.com/Lightning-AI/torchmetrics) (v0.11) | Apache-2.0 |
| PartiPrompts | `bigfix/metrics/PartiPrompts.tsv` | [google-research/parti](https://github.com/google-research/parti) | Apache-2.0 |
| PRISM-Bench prompts | `bigfix/metrics/PrismBenchPrompts.tsv` | FLUX-Reason-6M & PRISM-Bench | Apache-2.0 |

---

## 1. Code

### LlamaGen (MIT)

- **Files:** `bigfix/network/vq_model.py` is modified from LlamaGen's `tokenizer/tokenizer_image/vq_model.py`.
  `top_k_top_p_filtering` in `bigfix/sampler/halton_sampler.py` follows LlamaGen's
  `autoregressive/models/generate.py`, which itself credits
  [this gist](https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317) by Thomas Wolf.
- **Source:** <https://github.com/FoundationVision/LlamaGen>

```text
MIT License

Copyright (c) 2024 FoundationVision

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### taming-transformers (MIT)

LlamaGen's tokenizer, from which `bigfix/network/vq_model.py` is modified, credits taming-transformers (and
[MaskGIT](https://github.com/google-research/maskgit) by Google Research, Apache-2.0) as its own sources.

- **Source:** <https://github.com/CompVis/taming-transformers>

```text
Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE./
```

### nanoGPT (MIT)

- **Files:** parts of `bigfix/network/transformer.py` are borrowed from nanoGPT.
- **Source:** <https://github.com/karpathy/nanoGPT>

```text
MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### prdc (MIT)

- **Files:** `compute_pairwise_distance`, `get_kth_value`, `compute_nearest_neighbour_distances` and `compute_prdc`
  in `bigfix/metrics/inception_metrics.py` are adapted from `prdc/prdc.py`, the implementation of
  *Reliable Fidelity and Diversity Metrics for Generative Models* (Naeem et al., ICML 2020).
- **Changes:** they are methods of the metric class here instead of module-level functions.
- **Source:** <https://github.com/clovaai/generative-evaluation-prdc>

```text
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### DiT (CC BY-NC 4.0)

- **Files:** the `modulate` function (`bigfix/network/transformer_block.py`, and a renamed-argument variant in
  `bigfix/network/txt_transformer.py`) and the adaptive-modulation layer structure of `AdaNorm`
  (`bigfix/network/transformer_block.py`) follow `models.py` of DiT, "Scalable Diffusion Models with Transformers".
- **Copyright:** (c) Meta Platforms, Inc. and affiliates.
- **License:** Creative Commons Attribution-NonCommercial 4.0 International,
  <https://creativecommons.org/licenses/by-nc/4.0/legalcode>. Please read its terms, in particular the
  NonCommercial condition, before reusing these parts.
- **Changes:** adapted to this project (renamed arguments, and `AdaNorm` uses RMSNorm).
- **Source:** <https://github.com/facebookresearch/DiT>

### torchmetrics (Apache-2.0)

- **Files:** `MatrixSquareRoot`, the FID computation and `inception_score` in `bigfix/metrics/inception_metrics.py`
  are adapted from `torchmetrics/image/fid.py` and `torchmetrics/image/inception.py` (torchmetrics v0.11).
  torchmetrics credits its matrix square root implementation to "Square Root of a Positive Definite Matrix".
- **Copyright:** The PyTorch Lightning team.
- **Changes:** the code was modified to compute FID, Inception Score and (optionally) precision, recall, density
  and coverage in a single metric object, and to resize images with a custom resizer.
- **License:** the Apache License 2.0, reproduced in full in section 3.
- **Source:** <https://github.com/Lightning-AI/torchmetrics>

## 2. Data

### PartiPrompts

`bigfix/metrics/PartiPrompts.tsv` contains the PartiPrompts benchmark released by Google Research with the paper
*Scaling Autoregressive Models for Content-Rich Text-to-Image Generation* (Yu et al., 2022). The
[google-research/parti](https://github.com/google-research/parti) repository is licensed under the Apache License 2.0
(section 3).

### PRISM-Bench

`bigfix/metrics/PrismBenchPrompts.tsv` contains the prompts of PRISM-Bench, from *FLUX-Reason-6M & PRISM-Bench: A
Million-Scale Text-to-Image Reasoning Dataset and Comprehensive Benchmark* (arXiv:2509.09680). The FLUX-Reason-6M
[dataset card](https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M) lists the Apache License 2.0
(section 3). The prompts are stored in a tab-separated file here.

## 3. Apache License 2.0

This is the full text of the license that applies to the torchmetrics, PartiPrompts and PRISM-Bench material above.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2020-2022 Lightning-AI team

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## 4. Models and libraries that are not redistributed

This repository does not include any pretrained weights or third-party libraries. The following are downloaded
or installed by the user from their original sources, and remain under their own licenses and terms of use:

- the pretrained VQGAN checkpoints from [LlamaGen](https://huggingface.co/FoundationVision/LlamaGen)
  (class-to-image) and from the [BIGFix Hugging Face repository](https://huggingface.co/llvictorll/BigFIX)
  (text-to-image);
- the text encoder `google/flan-t5-xl` and the CLIP, DINOv2 and PickScore models fetched through Hugging Face
  `transformers`;
- the optional reward backends installed with `pip install -e ".[reward]"` (HPSv2, ImageReward,
  aesthetic-predictor);
- the Python dependencies listed in [pyproject.toml](pyproject.toml).
