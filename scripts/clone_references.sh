#!/usr/bin/env bash
# Clone upstream reference repos for comparison with integrated code.
# Run once on a new machine: bash scripts/clone_references.sh

set -e
cd "$(dirname "$0")/.."

mkdir -p references

if [ ! -d references/egnn ]; then
    echo "Cloning EGNN..."
    git clone --depth 1 https://github.com/vgsatorras/egnn.git references/egnn
fi

if [ ! -d references/cpmp ]; then
    echo "Cloning CPMP..."
    git clone --depth 1 https://github.com/panda1103/CPMP.git references/cpmp
fi

if [ ! -d references/se3-transformer-nvidia ]; then
    echo "Cloning NVIDIA SE3-Transformer (sparse)..."
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/NVIDIA/DeepLearningExamples.git references/se3-transformer-nvidia
    cd references/se3-transformer-nvidia
    git sparse-checkout set DGLPyTorch/DrugDiscovery/SE3Transformer
    cd ../..
fi

echo "Done. References:"
du -sh references/*/
