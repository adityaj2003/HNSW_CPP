# HNSW C++ Implementation

This is a CPP implementation of the HNSW paper. I wrote it for use with my semantic search PDF app repo. The select neighbors implemented is the simple version.

📄 **Paper URL**: [Hierarchical Navigable Small World Graphs (HNSW)](https://arxiv.org/pdf/1603.09320)

## Overview

This implementation compares **Hierarchical Navigable Small World (HNSW) search** with a **brute-force linear search**. The main function:
- Generates **1000 random 384-dimensional embeddings**.
- Inserts them into **HNSW** and **performs K-NN searches**.
- Runs a **linear search** to find exact nearest neighbors.
- Compares results in terms of **speed and accuracy**.
- I also have a separate file for writing index to disk using mmap
At **1000 embeddings**, HNSW provides **~3x faster search time** than brute-force search, but **construction time is significantly longer**. This is a **tradeoff between indexing time and recall**, but for my use case, I prefer greater indexing time to optimize search speed.
Also mmap takes about the same amount of time to load and search from it. I
tried offloading to disk so that the hnsw indexing does not have to run
everytime we open a file. Can try a notetaking app for indexing files and
building an automatic graph generation using keywords.
## Future Improvements
I intend to add **concurrent insertion** in a future version if possible, to improve indexing speed without sacrificing search efficiency.

