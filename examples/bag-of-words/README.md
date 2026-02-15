# kornia-bow Feature Demo

This example demonstrates the core functionality of the `kornia-bow` crate, including:

1.  **Data Generation**: Creating synthetic binary descriptors (simulating features like ORB or BRIEF).
2.  **Vocabulary Training**: Building a hierarchical vocabulary tree using KMeans++.
3.  **I/O Operations**: Saving and loading the vocabulary to disk.
4.  **Transformation**: Converting sets of image features into Bag-of-Words (BoW) vectors.
5.  **Scoring**: Computing similarity scores between BoW vectors using multiple metrics:
    - L1 Norm
    - L2 Norm
    - Chi-Square
    - KL Divergence
    - Bhattacharyya
    - Dot Product
6.  **Direct Index**: Generating a direct index for geometric verification.

## Running the Example

From the root of the repository:

```bash
cargo run -p bag-of-words
```
