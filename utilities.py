from typing import TypeVar, Callable, Iterable, List, Union, Tuple
from joblib import Parallel, delayed

A = TypeVar('A')  # Base input type before expanding to tuple
R = TypeVar('R')  # return type

def parallel_map(
    func: Callable[..., R],
    iterable: Iterable[A],
    expand_args: Callable[[A], Tuple],
    n_jobs: int = -1,
    backend: str = 'multiprocessing',
    verbose: int = 0
) -> List[R]:
    """
    Maps a function over an iterable in parallel, returning only successful results.
    Each input value is first expanded into a tuple of arguments.
    
    Args:
        func: The function to apply
        iterable: The input iterable
        expand_args: Function to convert each input into a tuple of arguments
        n_jobs: Number of parallel jobs (-1 for all cores)
        backend: Joblib backend ('multiprocessing' or 'threading')
        verbose: Verbosity level for joblib
    """
    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(func)(*expand_args(x)) for x in iterable
    )