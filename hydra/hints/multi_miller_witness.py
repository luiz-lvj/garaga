import time
import sympy
from hydra.algebra import PyFelt, Polynomial
from hydra.definitions import (
    tower_to_direct,
    direct_to_tower,
    CURVES,
    CurveID,
    G1Point,
    G2Point,
)
import math
from hydra.hints.tower_backup import E12


def get_m_dash_root(f: E12) -> tuple[int, int, int, int, int, int]:
    assert f.curve_id == CurveID.BN254.value

    q = CURVES[f.curve_id].p
    x = CURVES[f.curve_id].x
    r = CURVES[f.curve_id].n

    h = (q**12 - 1) // r  # = 3^3 · l # where gcd(l, 3) = 1
    assert math.gcd(r, h) == 1
    base = 3
    k, l = decompose_scalar_into_b_powers_and_remainder(h, base)
    assert base**k * l == h, f"{base}^k * l should be h"
    assert h % (base**k) == 0, f"h should be a multiple of {base}^k"
    assert math.gcd(l, base) == 1, f"l should be coprime with {base}"

    λ = (
        6 * x + 2 + q - q**2 + q**3
    )  # https://eprint.iacr.org/2008/096.pdf See section 4 for BN curves.

    assert λ % r == 0, "λ should be a multiple of r. See section 4.2.2"
    m = λ // r
    d = math.gcd(m, h)

    assert m % d == 0, "m should be a multiple of d"
    m_dash = m // d  # m' = m/d
    assert m_dash % h != 0
    f"m/d should not divide h. See section 4.2.2 Theorem 2."
    assert d * r * m_dash == λ, "incorrect parameters"  # sanity check
    assert math.gcd(λ, q**12 - 1) == d * r
    assert (
        math.gcd(m_dash, q**12 - 1) == 1
    ), f"m_dash should be coprime with q**12 - 1 'by construction'. See 4.3.2 computing m-th root"
    m_d_inv = pow(m_dash, -1, h)

    return f**m_d_inv


def decompose_scalar_into_b_powers_and_remainder(scalar: int, b: int):
    """
    Decompose scalar into b^k * l, where l is not divisible by b.
    """

    k = 0
    l = scalar

    while l % b == 0:
        l //= b
        k += 1
    assert l % b != 0, "l should not be divisible by b"
    assert scalar == b**k * l, "scalar should be the product of b^k * l"
    return k, l


def pow_3_ord(a: E12):
    t = 0
    FP12_ONE = E12.one(a.curve_id)
    while a != FP12_ONE:
        t += 1
        a = a**3
    return t


def find_cube_root(a: E12, w: E12, q: int) -> E12:
    # Algorithm 4: Modified Tonelli-Shanks for cube roots
    # Input: Cube residue a, cube non residue w and write p − 1 = 3^r · s such that 3 ∤ s
    # Output: x such that x^3 = a
    # 1 exp = (s + 1)/3
    _, s = decompose_scalar_into_b_powers_and_remainder(q**12 - 1, 3)
    exp: int = (s + 1) // 3
    a_inv = a.__inv__()
    # 2 x ← a^exp
    x = a**exp
    # 3 3^t ← ord((x^3)/a)
    t = pow_3_ord(x**3 * a_inv)
    # 4 while t != 0 do
    while t != 0:
        # 5 exp = (s + 1)/3
        # 6 x ← x · w^exp
        x = x * w**exp
        # 7 3^t ← ord(x^3/a)
        t = pow_3_ord(x**3 * a_inv)
    # 8 end
    # 9 return x
    return x


def find_c_e12(f: E12, w: E12):
    # Algorithm 5: Algorithm for computing λ residues over BN curve
    # Input: Output of a Miller loop f and fixed 27-th root of unity w
    # Output: (c, wi) such that c**λ = f · wi
    # 1 s = 0
    s = 0
    q = CURVES[f.curve_id].p
    exp = (q**12 - 1) // 3
    FP12_ONE = E12.one(f.curve_id)
    # 2 if f**(q**k-1)/3 = 1 then
    if f**exp == FP12_ONE:
        # 3 continue
        # 4 end
        # 5 else if (f · w)**(q**k-1)/3 = 1 then
        c = f
    elif (f * w) ** exp == FP12_ONE:
        # 6 s = 1
        s = 1
        # 7 f ← f · w
        c = f * w
    # 8 end
    # 9 else
    else:
        # 10 s = 2
        s = 2
        # 11 f ← f · w**2
        c = f * w * w
    # 12 end
    # 13 c ← f**r′
    c = get_rth_root(c)
    # 14 c ← c**m′′
    c = get_m_dash_root(c)
    # 15 c ← c**1/3 (by using modified Tonelli-Shanks 4)
    c = find_cube_root(c, w, q)
    # 16 return (c, ws)
    return c, w**s


def get_rth_root(f: E12) -> E12:
    """
    Computes x such that x^r = f
    """
    r = CURVES[f.curve_id].n
    h = (CURVES[f.curve_id].p ** 12 - 1) // r
    r_inv = pow(r, -1, h)
    res = f**r_inv
    assert res**r == f, "res**r should be f"
    return res


def get_27th_bn254_root():
    """
    Retrieve a 27th root of unity over BN254 Fp^12 that isn't a 9th root of unity.
    """
    root_27th = E12(
        [
            0,
            0,
            0,
            0,
            8204864362109909869166472767738877274689483185363591877943943203703805152849,
            17912368812864921115467448876996876278487602260484145953989158612875588124088,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        CurveID.BN254.value,
    )
    FP12_ONE = E12.one(CurveID.BN254.value)
    assert root_27th**27 == FP12_ONE, "root_27th**27 should be one"
    assert root_27th**9 != FP12_ONE, "root_27th**9 should not be one"
    return root_27th


if __name__ == "__main__":

    from tools.gnark_cli import GnarkCLI

    def get_miller_loop_output(curve_id: CurveID) -> E12:
        """
        Returns a random miller loop output f such that f**h = 1
        """
        cli = GnarkCLI(curve_id=curve_id)
        g1, g2 = G1Point.gen_random_point(curve_id), G2Point.gen_random_point(curve_id)
        neg_g1 = -g1
        # Miller (-g1, g2) * Miller (g1, g2)
        f: E12 = cli.miller(
            [
                neg_g1.x,
                neg_g1.y,
                g2.x[0],
                g2.x[1],
                g2.y[0],
                g2.y[1],
                g1.x,
                g1.y,
                g2.x[0],
                g2.x[1],
                g2.y[0],
                g2.y[1],
            ],
            2,
            raw=False,
        )
        h = (CURVES[curve_id.value].p ** 12 - 1) // CURVES[curve_id.value].n
        assert f**h == E12.one(curve_id.value), "f**h should be one"
        return f

    def test_bn254():
        f = get_miller_loop_output(CurveID.BN254)
        x = CURVES[CurveID.BN254.value].x
        q = CURVES[CurveID.BN254.value].p
        λ = (
            6 * x + 2 + q - q**2 + q**3
        )  # https://eprint.iacr.org/2008/096.pdf See section 4 for BN curves.

        c, wi = find_c_e12(f, get_27th_bn254_root())
        c_inv = c.__inv__()
        result = c_inv**λ * f * wi
        assert result == E12.one(CurveID.BN254.value), "pairing not 1"

    def test_bls12_381():
        x = CURVES[CurveID.BLS12_381.value].x
        q = CURVES[CurveID.BLS12_381.value].p
        r = CURVES[CurveID.BLS12_381.value].n

        k = ((x - 1) ** 2) // 3
        # r = (q-x)//k by construction for bls:
        assert q == k * r + x

        f = get_miller_loop_output(CurveID.BLS12_381)
        c = get_rth_root(f)

        # Only theorem 1 can be applied with bls constants:
        assert f == c**r

        # But since r = (q-x)//k, we have:
        # f = c**r
        # <=> f = (c**(q-x))**(1//k)
        # <=> f^k = ((c**(q-x))**(1//k))^k
        # <=> f^k = (c**(q-x))^(k//k)
        # <=> f^k = c^(q-x)

        # f/c^(-x) can be computed easily within the miller loop with free squarings.
        # c^q is virtually free to compute with one Frobenius, therefore we can obtain f/c^(q-x) easily.
        # k is only 126 bits, providing ~ 50% reduction in cost compared to a full final exponentiation.
        assert f**k == c ** (q - x)

    for i in range(10):
        test_bn254()
        test_bls12_381()
        print(f"Test {i} passed")