
<br>
<p align="center">
<h1 align="center"><strong> Coordinating Search with Foundation Models and Multi-Agent Reinforcement Learning in Complex Environments</strong></h1>
  <p align="center">
    <a href='https://www.usudirectlab.org' target='_blank'>Direct Lab</a>&emsp;
    <br>
    Department of Computer Science, Utah State University, Nvidia,  Autonomous Systems Branch, Army Research Lab (ARL), Department of Computer Science, University of Washington
    <br>
  </p>
</p>


<!-- UPDATE THIS -->
[![](https://img.shields.io/badge/Paper-%F0%9F%93%96-yellow)](https://example.com/paper-placeholder)
[![](https://img.shields.io/badge/Project-%F0%9F%9A%80-pink)](https://github.com/Zenif-NIght/Coordinatig-MAT-Env)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)
[![](https://img.shields.io/badge/Documentation-📘-green)](Docs/getting_started.md)
[![](https://img.shields.io/badge/Youtube-🎬-red)](https://www.youtube.com/watch?v=your-video-id) 



This repository contains the codebase for our research on coordinating search operations using multi-agent reinforcement learning (MARL) and foundation models in complex, unstructured environments.

## Overview

Our system simulates a fleet of heterogeneous robots designed for search operations in challenging outdoor scenarios. The approach combines reinforcement learning (RL) and a centralized Multi-Agent Transformer (MAT) to coordinate autonomous agents, integrating sensory data into a structured knowledge graph for efficient decision-making.

### Key Features

- **Multi-Agent Reinforcement Learning**: Autonomous coordination among diverse robots using MARL.
- **Multi-Agent Transformer (MAT)**: Centralized decision-making model for high-level command interpretation.
- **Simulation Environment**: Developed with NVIDIA Isaac Sim and GRUtopia for high-fidelity and scalable robot simulations.


## Installation

CODE <COMING SOON>

## Usage

COMING SOON!

## Complex Environments

We evaluate our approach in complex, unstructured environments to test the system's robustness and scalability. The environments include:


* Rock fields (Nvidia Blast)
  
  A giude on how to create and use a destructible Environments with Blast is available [here](Docs/set_up_blast.md)

  ![Blast](Docs/assets/blast/blast.gif)

* Forest (Nvidia optimized vegetation assets)

  You can download a forest environment [here](https://usu-my.sharepoint.com/:f:/g/personal/a02233404_aggies_usu_edu/EiaLwVKnKUJIhuea4IRn7G8BSAsc93-9ItGP3WBFFrigmw?e=0rFzQC)

  ![Forest](Docs/assets/envs/tile_forest1.png)
  
* Night and Day Lighting (Nvidia Sun Study)

These environments are open-source and available for download SOON

### Downloading the Environments

<COMING SOON>


## Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](Docs/CONTRIBUTING.md) for details on the code of conduct and submission process.

## License

The project License is found in the [LICENSE](LICENSE)

## Acknowledgments

This work was supported by the U.S. Army Research Fellowship Program and the Department of Computer Science, Utah State University.
- [MAT](https://github.com/PKU-MARL/Multi-Agent-Transformer): We use MAT for centralized decision-making in the multi-agent system.
- [GRUtopia](https://github.com/OpenRobotLab/GRUtopia): We use GRUtopia for simulation of the robots.
    ### GRUtopia Acknowledgments
    - [OmniGibson](https://github.com/StanfordVL/OmniGibson): GRUtopia refers to OmniGibson for designs of oracle actions.
    - [RSL_RL](https://github.com/leggedrobotics/rsl_rl): GRUtopia utilized `rsl_rl` library to train the control policies for legged robots.
    - [ReferIt3D](https://github.com/referit3d/referit3d): GRUtopia refers to the Sr3D's approach to extract spatial relationship.
    - [Isaac Lab](https://github.com/isaac-sim/IsaacLab): GRUtopia utilized some utilities from Orbit (Isaac Lab) for driving articulated joints in Isaac Sim.



## Citation

If you find this work helpful, please cite it as follows:

```
@inproceedings{allred2024coordinating,
  title={Coordinating Search with Foundation Models and Multi-Agent Reinforcement Learning in Complex Environments},
  author={Allred, Christopher and Haight, Jacob and Justice, Chandler and Peterson, Isaac and Scalise, Rosario and Hromadka, Ted and Pusey, Jason and Harper, Mario},
    year={2024},
}
```

