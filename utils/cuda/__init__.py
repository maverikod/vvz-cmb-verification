"""
CUDA acceleration utilities for CMB verification project.

Provides block-based array processing, vectorization, and GPU acceleration
for computationally intensive operations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from utils.cuda.array_model import CudaArray, CudaUnavailableError, CudaMemoryError
from utils.cuda.vectorizer import BaseVectorizer
from utils.cuda.elementwise_vectorizer import ElementWiseVectorizer
from utils.cuda.transform_vectorizer import TransformVectorizer
from utils.cuda.reduction_vectorizer import ReductionVectorizer
from utils.cuda.correlation_vectorizer import CorrelationVectorizer
from utils.cuda.grid_vectorizer import GridVectorizer

__all__ = [
    "CudaArray",
    "CudaUnavailableError",
    "CudaMemoryError",
    "BaseVectorizer",
    "ElementWiseVectorizer",
    "TransformVectorizer",
    "ReductionVectorizer",
    "CorrelationVectorizer",
    "GridVectorizer",
]
