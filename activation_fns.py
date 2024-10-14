import jax
import jax.numpy as jnp
import numpy as np

EPSILON = 1e-7
activation_fns_list = [
    lambda x: jnp.float32(x),  # identity
    lambda x: 1 / (1 + jnp.exp(-x)),  # sigmoid
    lambda x: jnp.divide(1, x + EPSILON),  # inverse (with epsilon)
    lambda x: jnp.tanh(x),  # hyperbolic tangent
    lambda x: jnp.float32(jnp.maximum(0, x)),  # relu
    lambda x: jnp.float32(jnp.abs(x)),  # absolute value
    lambda x: jnp.sin(x),  # sine
    lambda x: jnp.exp(-jnp.square(x)),  # gaussian
    lambda x: jnp.float32(x >= 0),  # step
]

def get_activation_fn(activation_index: int, x: float) -> jnp.float32:
    if activation_index < 0 or activation_index >= len(activation_fns_list):
        raise ValueError(f"Invalid activation index: {activation_index}")
    return jax.lax.switch(
        activation_index,
        activation_fns_list,
        operand=x,
    )

def test_identity():
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    result = jax.vmap(lambda x: get_activation_fn(0, x))(x)
    expected = x
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("Identity test passed")

def test_sigmoid():
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    result = jax.vmap(lambda x: get_activation_fn(1, x))(x)
    expected = 1 / (1 + np.exp(-x))
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("Sigmoid test passed")

def test_inverse():
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    result = jax.vmap(lambda x: get_activation_fn(2, x))(x)
    expected = 1 / (x + EPSILON)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("Inverse test passed")

def test_tanh():
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    result = jax.vmap(lambda x: get_activation_fn(3, x))(x)
    expected = np.tanh(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("Hyperbolic tangent test passed")

def test_relu():
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    result = jax.vmap(lambda x: get_activation_fn(4, x))(x)
    expected = np.maximum(0, x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("ReLU test passed")

def test_abs():
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    result = jax.vmap(lambda x: get_activation_fn(5, x))(x)
    expected = np.abs(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("Absolute value test passed")

def test_sin():
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    result = jax.vmap(lambda x: get_activation_fn(6, x))(x)
    expected = np.sin(x)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("Sine test passed")

def test_gaussian():
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    result = jax.vmap(lambda x: get_activation_fn(7, x))(x)
    expected = np.exp(-np.square(x))
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("Gaussian test passed")

def test_step():
    x = jnp.array([-1.0, 0.0, 1.0, 2.0])
    result = jax.vmap(lambda x: get_activation_fn(8, x))(x)
    expected = (x >= 0).astype(np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("Step function test passed")

def test_invalid_index():
    try:
        get_activation_fn(-1, 0.0)
    except ValueError as e:
        print(f"Invalid index test passed: {e}")
    else:
        raise AssertionError("Invalid index test failed")

    try:
        get_activation_fn(9, 0.0)
    except ValueError as e:
        print(f"Invalid index test passed: {e}")
    else:
        raise AssertionError("Invalid index test failed")

if __name__ == "__main__":
    test_identity()
    test_sigmoid()
    test_inverse()
    test_tanh()
    test_relu()
    test_abs()
    test_sin()
    test_gaussian()
    test_step()
    test_invalid_index()
    print("All tests passed!")
