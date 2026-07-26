# RNGDetPP-X
Density-based specialization of RNGDet++ for road network extraction from high-resolution aerial imagery. Implements hard routing across specialized expert models by road density (3 classes: low-rural, medium-suburban, and high-urban).
# RNGDetPP-X

**Density-specialized extension of [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus) for road network graph detection from high-resolution aerial imagery.**

RNGDetPP-X extends the original RNGDet++ with **density-based specialization**: a CNN density classifier performs **hard routing** of image patches to three expert models trained separately for **rural**, **suburban**, and **urban** road densities. The approach is evaluated on the **RSEB** dataset (orthoimagery from southern Brazil, 2048×2048 px tiles).

This repository is part of a master's thesis developed at the Graduate Program in Computing (PPGC), Institute of Informatics, Federal University of Rio Grande do Sul (UFRGS). It builds directly on the official RNGDet++ implementation by Xu et al. (2023); see [Credits](#credits).

> ⚠️ **License notice:** RNGDet++ is released under **GPL-3.0** ("only for academic research, not for commercial purposes"). As a derivative work, this repository **must** also be licensed under GPL-3.0. See [License](#license).

---

## Motivation

Road geometry and topology vary widely across land-use contexts: a single generalist model must reconcile sparse rural roads with dense, grid-like urban networks in one set of weights. RNGDetPP-X investigates whether **specializing models by road density** improves extraction quality over a single generalist model, following a Mixture-of-Experts rationale with **hard routing** instead of a learned soft gating.

The pipeline is:

1. A **ResNet-50** classifier assigns each 128×128 px patch to one of three density classes (rural, suburban, urban).
2. The patch is routed to the corresponding **expert model** (RNGDet++ agent with ResNet backbone), fine-tuned only on samples of that class.
3. The iterative Transformer-based agent reconstructs the road network graph vertex by vertex.

---

## Experimental protocol

We explore in this repository only the Experiment 3. Three experimental scenarios, matching the thesis:

| Experiment | Description |
|------------|-------------|
| **Experiment 1** | Zero-shot inference of RNGDet++ pre-trained on CityScale, applied directly to RSEB (baseline). |
| **Experiment 2** | Generalist fine-tuning of a single RNGDet++ model on RSEB. |
| **Experiment 3** | Density-specialized experts with hard routing (classifier + 3 expert models). |

**Evaluation metrics:** APLS, TOPO (Precision, Recall, F1).

---

## Repository structure

The structure follows the original RNGDet++ repository. Items marked **`# NEW`** are contributions of this work (the "-X" extension); adjust the names below to match your actual files.

```
RNGDetPP-X
├── cityscale/                # CityScale pipeline: sampler, train, test, metrics (from RNGDet++)
│    └── metrics/             # TOPO (topo.bash) and APLS (apls.bash)
├── spacenet/                 # SpaceNet pipeline (from RNGDet++)
├── rseb/                     # NEW: RSEB pipeline (sampler, train, test, metrics)
│    └── metrics/             # NEW: TOPO / APLS adapted for RSEB
├── docker/                   # Dockerfile + build_image.bash (from RNGDet++)
├── density_classifier/       # NEW: ResNet-50 density classifier (128×128 px)
├── routing/                  # NEW: hard-routing between classifier and experts
├── bash/                     # run_sampler / run_train / run_test scripts (from RNGDet++)
├── prepare_dataset/          # dataset + pretrained checkpoint preparation (from RNGDet++)
├── build_container_cityscale.bash
├── build_container_spacenet.bash
├── build_container_rseb.bash # NEW
├── LICENSE.md                # GPL-3.0 (inherited from RNGDet++)
└── README.md
```

---

## Platform & environment

RNGDet++ runs **inside a Docker container** (all steps except evaluation). This environment differs substantially from the SAM-Road-X / DeH4R-X repositories.

Reference environment (original repo):

```
Ubuntu 20.04, CUDA 11.1, Docker 20.10.7, Nvidia-driver 495.29.05
PyTorch 1.8 (inside the container)
```

> Thesis experiments were run on the lab GPU server (~27 h training per experiment). Report your own hardware here.

### Build the Docker image
```bash
cd docker
./build_image.bash
```

### Start a container
```bash
# set RNGDet_dir inside the script first
./build_container_rseb.bash        # NEW (adapted from build_container_cityscale.bash)
```

> ⚠️ **Evaluation metric scripts are NOT runnable inside Docker.** Run APLS/TOPO outside the container.

---

## Data & pretrained checkpoints

Prepare datasets and the RNGDet / RNGDet++ pretrained checkpoints:

```bash
cd prepare_dataset
./preprocessing.bash
```

> Note: in the original repo, the Google Drive links were replaced; data and checkpoints are distributed via a Tsinghua Cloud link (see the original README). Update paths for your setup.

The **RSEB** dataset consists of orthoimages from southern Brazil, tiled at 2048×2048 px, annotated for road network extraction. <!-- Add access / availability details (see data-sharing note). -->

---

## Usage

The RNGDet++ workflow is: **sample → train → infer → evaluate**, driven by `.sh`/`.bash` scripts.

### 1. Generate training samples
```bash
./bash/run_sampler.sh
```

### 2. Train the density classifier   `# NEW`
```bash
python density_classifier/train.py --config rseb/config/classifier_resnet50.yaml
```

### 3. Train the expert models (one per density class)   `# NEW`
```bash
./bash/run_train_expert_rural.sh
./bash/run_train_expert_suburban.sh
./bash/run_train_expert_urban.sh
```

### 4. Inference with hard routing (Experiment 3)   `# NEW`
```bash
python routing/infer_routed.py --config rseb/config/exp3_experts.yaml
```

For the baseline experiments, the original RNGDet++ entry points are used directly:

```bash
# Experiment 1 (zero-shot) — RNGDet++ pre-trained on CityScale, applied to RSEB
./bash/run_test_RNGDet++.sh

# Experiment 2 (generalist fine-tuning on RSEB)
./bash/run_train_RNGDet++.sh
```

> Key inference parameters (from RNGDet++): `logit_threshold` (0.75), `candidate_filter_threshold` (30), `extract_candidate_threshold` (0.7), `alignment_distance` (5), plus flags `instance_seg` and `multi_scale`. Enable `instance_seg` and `multi_scale` to reproduce full RNGDet++ (vs. base RNGDet).

### 5. Evaluation (outside Docker)
```bash
cd rseb/metrics
./topo.bash        # TOPO
./apls.bash        # APLS (requires Go)
```

---

## Related repositories

This thesis evaluates three architectures, each with its own density-specialized extension:

- **RNGDetPP-X** — this repository
- **[SAM-Road-X](https://github.com/)** <!-- add URL --> — SAM-Road with density specialization (ConvNeXt-Tiny classifier, 512×512 px)
- **[DeH4R-X](https://github.com/)** <!-- add URL --> — DeH4R with density specialization

---

## Credits

This work is a derivative of the official RNGDet++ implementation (GPL-3.0):

> Xu, Z.; Liu, Y.; Sun, Y.; Liu, M.; Wang, L. **RNGDet++: Road Network Graph Detection by Transformer with Instance Segmentation and Multi-scale Features Enhancement.** IEEE Robotics and Automation Letters, v. 8, n. 5, p. 2991–2998, 2023. Original code: https://github.com/TonyXuQAQ/RNGDetPlusPlus

RNGDet++ itself builds on Sat2Graph and DETR.

---

## Citation

If you use this code, please cite both the thesis and the original RNGDet++ paper.

**Thesis:**
```bibtex
@mastersthesis{rngdetppx,
  author  = {<Author>},
  title   = {Extração de Redes Viárias a Partir de Imagens Aéreas: Uma Abordagem com Modelos Especializados em Densidade Urbana e Inferência Iterativa Baseada em Grafos},
  school  = {Universidade Federal do Rio Grande do Sul, Programa de Pós-Graduação em Computação},
  year    = {2026},
  address = {Porto Alegre, Brasil}
}
```

**RNGDet++ (original method):**
```bibtex
@article{xu2023rngdet++,
  title={RNGDet++: Road Network Graph Detection by Transformer with Instance Segmentation and Multi-scale Features Enhancement},
  author={Xu, Zhenhua and Liu, Yuxuan and Sun, Yuxiang and Liu, Ming and Wang, Lujia},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={5},
  pages={2991--2998},
  year={2023},
  publisher={IEEE}
}
```

---

## License

This repository is licensed under the **GNU General Public License v3.0 (GPL-3.0)**, inherited from the original RNGDet++ codebase. Per the original terms, use is restricted to **academic research and non-commercial purposes**.

> Note: given the ties to the Brazilian Army (DSG/CGEOs), confirm with your advisor that the intended use falls within the "academic research, non-commercial" scope required by the GPL-3.0 license of the upstream RNGDet++ code, and confirm the **data-sharing policy** for the RSEB dataset before making the repository or data public.

---

## Acknowledgments

Developed at PPGC/UFRGS under the supervision of Prof. Dr. Claudio Rosito Jung, with Prof. Dr. Anderson Rocha Tavares as co-supervisor.
