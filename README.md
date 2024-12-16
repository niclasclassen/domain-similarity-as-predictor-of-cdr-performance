# Domain Similarity as a Predictor of Cross-Domain Recommendation Performance

## Abstract

This study investigates the potential correlation between domain similarity and negative transfer in cross-domain recommender systems. Two approaches are employed to measure domain similarity: one based on semantic similarity and the other on rating pattern similarity. Using these measures, we examine whether statistically significant differences in performance can be linked to the degree of similarity between domains. Our findings do not provide sufficient evidence to establish a statistically significant correlation. However, we argue that further research is essential to draw definitive conclusions.  

---

## Reproducibility

This repository provides all necessary resources and instructions to reproduce our experiments.

### Data

We use the following datasets for our experiments:

- - [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)

### Data Processing

For the data processing we use the following files:

- `run_download_and_process_data.py`
- `run_data_processing.py`
- `run_semantic_similarity_estimation.py`
- `run_semantic_ipynb_estimation.ipynb`

### Run Recommender Systems

We utilize the Recbole library for the implementation of recommender systems. Specifically, we employ:

- **recbole-cdr**:
- **recbole-debias**:

Both are modified to suit our experiments. These customized versions of both Recbole sub-libraries are included in the repository under:

- `vendor/recbole_cdr`
- `vendor/recbole_debias`

#### Dockerized Experiments

The models are containerized for ease of reproduction and to make them compatibility with a high-performance computing (HPC) clusters that we utilized. Docker images for our project are hosted on DockerHub:

- [niclasclassen/research-project-recbole-cdr-user-specific](https://hub.docker.com/r/niclasclassen/research-project-recbole-cdr-user-specific)
- [niclasclassen/research-project-recbole-debias](https://hub.docker.com/r/niclasclassen/research-project-recbole-debias)

#### Singularity Images for HPC

To ensure compatibility with the HPC, we convert the Docker images into Singularity images. The example job script, `example.job`, demonstrates how to execute experiments on an HPC environment.

## Results

The results of our experiments can be found in our report. The report is available in the repository under `report/report.pdf`.
