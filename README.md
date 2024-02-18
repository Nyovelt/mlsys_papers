# [WIP] mlsys_papers

[Github Pages](https://nyovelt.github.io/mlsys_papers/)

A curated list of machine learning papers from recent major system conferences, specifically SOSP and OSDI within the last three years, and high popularity paper. Topics, titles, keywords, and authors that are bolded reflect personal preferences or special relationship to us.

Topics of interest

## System for Machine Learning

### Distributed Systems

#### Model training Framework

- [NSDI '24] Parcae: Proactive, Liveput-Optimized DNN Training on Preemptible Instances [[link]](https://www.cs.cmu.edu/~zhihaoj2/papers/parcae.pdf)
  - Parcae [NSDI'24] is a system for cheap, fast, and scalable LLM training on spot instances. The key idea is a _proactive_, liveput-optimized approach to boosting preemption-aware throughput, reducing cost by 10x.
- [SOSP '23] Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates [[link]](https://doi.org/10.1145/3600006.3613152)
  - Oobleck enables resilient distributed training of large DNN models with guaranteed _fault tolerance_.
- [Arxiv '23] LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers [[link]](https://doi.org/10.48550/arXiv.2310.03294)
- [MLSys '22] Pathways: Asynchronous Distributed Dataflow for ML
  - Pathways uses a sharded dataflow graph of asynchronous operators that consume and produce futures, and efficiently gang-schedules heterogeneous parallel computations on thousands of accelerators while coordinating data transfers over their dedicated interconnects.
- [OSDI '22] Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning [[link]](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf)
  - Alpa designs a number of _compilation_ passes to automatically derive efficient parallel execution plans at each parallelism level.
- [SC '21] Efficient large-scale language model training on GPU clusters using megatron-LM [[link]](https://doi.org/10.5281/zenodo.5181820)
  - In this paper, we show how tensor, pipeline, and data parallelism can be composed to scale to thousands of GPUs.
- [OSDI '18] **Ray**: A Distributed Framework for Emerging AI Applications [[link]](https://www.usenix.org/conference/osdi18/presentation/moritz)
  - Ray implements a unified interface that can express both task-parallel and actor-based computations, supported by a single dynamic execution engine.

#### Model Serving

##### Model Inference Framework

- [ASPLOS '24] SpotServe: Serving Generative Large Language Models
on Preemptible Instances [[link]](https://www.cs.cmu.edu/~zhihaoj2/papers/spotserve.pdf)
  - For serving, SpotServe [ASPLOS'24] is an LLM serving system on spot instances. It handles instance preemptions with dynamic parallelization, promises low tail latency, and reduces serving cost by 54%.
- [SOSP '23] Efficient Memory Management for Large Language Model Serving with PagedAttention (**vLLM**) [[link]](https://doi.org/10.1145/3600006.3613165)
  - vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage.
- [OSDI '23] AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving [[link]](https://www.usenix.org/system/files/osdi23-li-zhuohan.pdf)
  - AlpaServe determines an efficient strategy for placing and _parallelizing_ collections of large deep learning models across a distributed cluster.
- [Arxiv '23] S-LoRA: Serving Thousands of Concurrent LoRA Adapters [[link]](https://doi.org/10.48550/arXiv.2311.03285)
- [OSDI '22] Orca: A Distributed Serving System for Transformer-Based Generative Models [[link]](https://www.usenix.org/conference/osdi22/presentation/yu)
- [NSDI '24] Approximate Caching for Efficiently Serving Diffusion Models [[link]](https://doi.org/10.48550/arXiv.2312.04429)
- [Arxiv '23] SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills [[link]](https://doi.org/10.48550/arXiv.2308.16369)

##### Scheduling and Resource Management

- [SOSP '23] Paella: Low-latency Model Serving with Software-defined GPU Scheduling [[link]](https://doi.org/10.1145/3600006.3613163)
  - Co-designing the model compiler, local clients, and the scheduler to bypass the built-in GPU scheduler and enable software control of kernel execution order.
- [OSDI '23] Hydro: Surrogate-Based Hyperparameter Tuning Service in the Datacenter [[link]](https://www.usenix.org/system/files/osdi23-hu.pdf)
  - Hydro, a surrogate-based hyperparameter tuning service that optimizes tuning workloads in both the job-level and cluster-level granularities.
- [OSDI '23] Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators [[link]](https://www.usenix.org/system/files/osdi23-zhao.pdf)
  - Partition a computational _graph of DNN_ into multiple sub-graphs by abstracting away hardware architecture and assign resources to each sub-graph,

#### Fault Tolerance

- [SOSP '23] GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints [[link]](https://doi.org/10.1145/3600006.3613145)
  - Gemini, a distributed training system that enables fast failure recovery for large model training by _checkpointing_ to CPU memory of the host machines with much larger aggregated bandwidth.

### Accelerate and optimize Machine Learning

- Bagpipe: Accelerating Deep Recommendation Model Training
- gSampler: General and Efficient GPU-based Graph Sampling for Graph Learning
- EinNet: Optimizing Tensor Programs with Derivation-Based Transformations
- Welder: Scheduling Deep Learning Memory Access via Tile-graph
- Grinder: Analysis and Optimization for Dynamic Control Flow in Deep Learning
- Accelerating Large Language Model Decoding with Speculative Sampling

#### AI Compiler and Programming Languages

- PIT: Optimization of Dynamic Sparse Deep Learning Models via Permutation Invariant Transformation
- Optimizing Dynamic Neural Networks with Brainstorm

#### Parallelism

- Gradient Compression Supercharged High-Performance Data Parallel DNN Training

### Database and Storage

- SPFresh: Incremental In-Place Update for Billion-Scale Vector Search
- VBase: Unifying Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity

### GPU Arch

- UGACHE: A Unified GPU Cache for Embedding-based Deep Learning

## Machine Learning for Systems

### Resource Management

- Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling

### Reliability

- HEALER: Relation Learning Guided Kernel Fuzzing
- Generating Complex, Realistic Cloud Workloads using Recurrent Neural Networks
- LUMOS: Efficient Kernel Concurrency Testing using a Learned Coverage Predictor

## Related Resources

- [Zhihao's Twitter](https://twitter.com/JiaZhihao/status/1758682486916218990)
- [A reading list for machine learning systems - Jeongseob Ahn](https://jeongseob.github.io/readings_mlsys.html)
- [eecs598/tree/w24-genai](https://github.com/mosharaf/eecs598/tree/w24-genai)
