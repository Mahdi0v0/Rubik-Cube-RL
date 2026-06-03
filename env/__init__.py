"""Environment layer: single-cube env (pure stdlib) + vectorized env (torch).

Per the dependency rule, torch appears ONLY in ``env/vec_env.py`` within this
package; ``env/cube_env.py`` stays pure standard library.
"""
