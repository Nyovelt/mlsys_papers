# mlsys_papers

machine learning related paper list selected from recent major system conferences

The papers are from SOSP, OSDI.

Topics of interest

## System for Machine Learning

### Distributed Systems

#### Model training Framework

- **Ray**: A Distributed Framework for Emerging AI Applications
- Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates

#### Model Serving
##### Model Serving Framework

- Efficient Memory Management for Large Language Model Serving with PagedAttention (**vLLM**)

##### Scheduling and Resource Management
- Paella: Low-latency Model Serving with Software-defined GPU Scheduling
- Hydro: Surrogate-Based Hyperparameter Tuning Service in the Datacenter 
- Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators
- Beta: Statistical Multiplexing with Model Parallelism for Deep Learning Serving
#### Fault Tolerance

- GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints

### Accelerate and optimize Machine Learning

- Bagpipe: Accelerating Deep Recommendation Model Training
- gSampler: General and Efficient GPU-based Graph Sampling for Graph Learning
- EinNet: Optimizing Tensor Programs with Derivation-Based Transformations
- Welder: Scheduling Deep Learning Memory Access via Tile-graph
- Grinder: Analysis and Optimization for Dynamic Control Flow in Deep Learning

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
