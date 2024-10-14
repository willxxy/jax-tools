import numpy as np
import jax.numpy as jnp
import jax
from jax import random

@jax.jit
def roulette_jax(key, pArr):
    """Returns random index, with each choices chance weighted
    Args:
    key - (PRNGKey) - JAX random number generator key
    pArr - (jnp.array) - vector containing weighting of each choice [N]
    Returns:
    choice - (int) - chosen index
    """
    spin = random.uniform(key) * jnp.sum(pArr)
    cumsum = jnp.cumsum(pArr)
    choice = jnp.sum(spin > cumsum)
    return jnp.where(choice < len(pArr), choice + 1, len(pArr))


@jax.jit
def rankArray_jax(X):
    """Returns ranking of a list, with ties resolved by first-found first-order
    NOTE: Sorts descending to follow numpy conventions
    """
    tmp = jnp.argsort(X)
    return jnp.argsort(tmp)

@jax.jit
def bestIntSplit_jax(ratio, total):
    """Divides a total into integer shares that best reflects ratio
    Args:
    share - [1 X N ] - Percentage in each pile
    total - [int ] - Integer total to split
    Returns:
    intSplit - [1 x N ] - Number in each pile
    """
    # Handle poorly defined ratio
    ratio = ratio / jnp.sum(ratio)
    # Get share in real and integer values
    floatSplit = ratio * total
    intSplit = jnp.floor(floatSplit)
    remainder = jnp.round(total - jnp.sum(intSplit)).astype(jnp.int32)
    # Rank piles by most cheated by rounding
    deserving = jnp.argsort(-(floatSplit-intSplit), axis=0)
    
    # Create a mask for the indices to update
    update_mask = jnp.arange(len(ratio)) < remainder
    
    # Use the mask to add 1 to the most deserving indices
    intSplit = intSplit + update_mask[deserving]
    
    return intSplit
