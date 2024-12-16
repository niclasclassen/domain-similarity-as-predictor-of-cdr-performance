# Domain Similarity as a Predictor of Cross-Domain Recommendation Performance

## Abstract

In this study we try to determine whether there exists a correlation between domain similarity and negative transfer in Cross Domain Recommender systems. We do this by employing two distinct approaches, one relying on semantical similarity of domains and the other relying on rating based pattern similarity of the domains. For these approaches we then investigate whether there exists a statistically significant difference in the performances that is correlated with the degree of similarity. Ultimately we find no grounds for accurately determining statistically significant results indicating a correlation. But we believe further research is worthwhile and required in order to make a final conclusion.

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
