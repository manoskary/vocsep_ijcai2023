# Voice Separation as Link Prediction - IJCAI 2023

This repository contains the code for the paper :
[Musical Voice Separation as Link Prediction: Modeling a Musical Perception Task as a Multi-Trajectory Tracking Problem]()

##### Abstract

This paper targets the perceptual task of separating the different interacting voices, i.e., monophonic melodic streams, in a polyphonic musical piece.  We target symbolic music, where notes are explicitly encoded, and model this task as a Multi-Trajectory Tracking (MTT) problem from discrete observations, i.e., notes in a pitch-time space. Our approach builds a graph from a musical piece, by creating one node for every note, and separates the melodic trajectories by predicting a link between two notes if they are consecutive in the same voice/stream.  This kind of local, greedy prediction is made possible by node embeddings created by a heterogeneous graph neural network that can capture inter- and intra-trajectory information. Furthermore, we propose a new regularization loss that encourages the output to respect the MTT premise of at most one incoming and one outgoing link for every node, favouring monophonic (voice) trajectories; this loss function might also be useful in other general MTT scenarios.
Our approach does not use domain-specific heuristics, is scalable to longer sequences and a higher number of voices, and can handle complex cases such as voice inversions and overlaps. We reach new state-of-the-art results for the voice separation task in classical music of different styles.

## Install and Run

It is suggested to create an environment using conda or miniconda more information [here](https://docs.conda.io/projects/miniconda/en/latest/index.html).
If you have conda then do:
```shell
conda create -n vocsep python=3.8
conda activate vocsep
```
The suggested version of python is 3.8 or later. To install requirements and run just do:
```shell
pip install -r requirements.txt
```
The requirements installs the cpu version of Pytorch and Pytorch-Scatter
If you want to run the code with CUDA please install the corresponding version of [Pytorch](https://pytorch.org/) and [Pytorch-Scatter](https://github.com/rusty1s/pytorch_scatter) to your system (follow links for more information).

To run the code just do:
```shell
python -m main.py
```

For more information about the configuration you can add execute with the flag `-h`.

This project depends on [WANDB](https://wandb.ai/) therfore you will need an account to run the code.

#### Results 

|            |       | **McLeod**    |       |       | **GMTT**      |       |       | **GMMT+LA**      |                 |
|------------|-------|---------------|-------|-------|---------------|-------|-------|------------------|-----------------|
| Datasets   | P     | R             | F1    | P     | R             | F1    | P     | R                | F1              |
| Inventions | 0.992 | 0.991         | 0.992 | 0.989 | 0.997         | 0.995 | 0.996 | 0.995            | **0.997**       |
| Sinfonias  | 0.982 | 0.982         | 0.982 | 0.987 | 0.989         | 0.978 | 0.987 | 0.982            | **0.985**       |
| WTC I      | 0.964 | 0.964         | 0.964 | 0.949 | 0.983         | 0.967 | 0.980 | 0.973            | **0.976**       |
| WTC II     | 0.964 | 0.964         | 0.964 | 0.945 | 0.979         | 0.962 | 0.976 | 0.968            | **0.972**       |
| Haydn      | 0.781 | 0.781         | 0.781 | 0.787 | 0.929         | 0.850 | 0.883 | 0.860            | **0.872**       |


For all results and model visit our logs at [WANDB](https://wandb.ai/vocsep/Voice%20Separation).
