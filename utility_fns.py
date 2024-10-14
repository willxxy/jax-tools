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

def listXor(b,c):
  """Returns elements in lists b and c they don't share
  """
  A = [a for a in b+c if (a not in b) or (a not in c)]
  return A

@jax.jit
def rankArray_jax(X):
    """Returns ranking of a list, with ties resolved by first-found first-order
    NOTE: Sorts descending to follow numpy conventions
    """
    tmp = jnp.argsort(X)
    return jnp.argsort(tmp)

def tiedRank(X):  
  """Returns ranking of a list, with ties recieving and averaged rank
  # Modified from: github.com/cmoscardi/ox_ml_practical/blob/master/util.py
  """
  Z = [(x, i) for i, x in enumerate(X)]  
  Z.sort(reverse=True)  
  n = len(Z)  
  Rx = [0]*n   
  start = 0 # starting mark  
  for i in range(1, n):  
     if Z[i][0] != Z[i-1][0]:
       for j in range(start, i):  
         Rx[Z[j][1]] = float(start+1+i)/2.0;
       start = i
  for j in range(start, n):  
    Rx[Z[j][1]] = float(start+1+n)/2.0;

  return jnp.asarray(Rx)

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

def quickINTersect(A, B):
    """ Faster set intersect: only valid for vectors of positive integers.
    (useful for matching indices)
    """
    if (len(A) == 0) or (len(B) == 0):
        return [], []
    P = np.zeros((1+max(max(A),max(B))), dtype=bool)
    P[A] = True
    IB = P[B]
    P[A] = False  # Reset
    P[B] = True
    IA = P[A]
    return IA, IB
