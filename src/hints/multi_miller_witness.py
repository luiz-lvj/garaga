import time
from src.algebra import PyFelt, Polynomial
from src.definitions import (
    tower_to_direct,
    direct_to_tower,
    get_base_field,
    CURVES,
    BN254_ID,
    BLS12_381_ID,
    CurveID,
    get_irreducible_poly,
)
import math
from src.hints.tower_backup import E12


field = get_base_field(BN254_ID)
irreducible_poly = get_irreducible_poly(curve_id=BN254_ID, extension_degree=12)


def int_to_felts(coeffs: list[int]) -> list[PyFelt]:
    return [field(x) for x in coeffs]


def int_to_e12(coeffs: list[int] | list[PyFelt]) -> E12:
    # call int_to_felts if coeffs is list of ints
    if type(coeffs[0]) != PyFelt:
        coeffs = int_to_felts(coeffs)
    return E12(coeffs, curve_id=BN254_ID)


def int_tower_to_direct(coeffs: list[int]) -> Polynomial:
    if type(coeffs[0]) != PyFelt:
        coeffs = int_to_felts(coeffs)
    return Polynomial(tower_to_direct(coeffs, curve_id=BN254_ID, extension_degree=12))


def e12_to_direct_poly(a: E12) -> Polynomial:
    return int_tower_to_direct(
        [
            a.c0.b0.a0,
            a.c0.b0.a1,
            a.c0.b1.a0,
            a.c0.b1.a1,
            a.c0.b2.a0,
            a.c0.b2.a1,
            a.c1.b0.a0,
            a.c1.b0.a1,
            a.c1.b1.a0,
            a.c1.b1.a1,
            a.c1.b2.a0,
            a.c1.b2.a1,
        ],
    )


import sympy


def is_prime(num: int) -> bool:
    return sympy.isprime(num)


def sample_xth_root(curve_id: int, x: int) -> E12:
    if x < 0:
        x_pos = -x
    r = E12.random(curve_id)
    p = CURVES[curve_id].p
    pow1 = p**12 - 1
    assert r**pow1 == E12.one(curve_id)
    pow2 = pow1 // x_pos
    root = r**pow2
    assert root**x_pos == E12.one(curve_id)
    root = root.__inv__() if x < 0 else root
    return root


xth_root = CURVES[BLS12_381_ID].x - 1
r = sample_xth_root(BLS12_381_ID, xth_root)
assert (r ** (-xth_root)).__inv__() == E12.one(BLS12_381_ID)


def build_params(
    curve_id: CurveID,
) -> tuple[int, int, int, int, int, int]:
    q = CURVES[curve_id.value].p
    x = CURVES[curve_id.value].x
    r = CURVES[curve_id.value].n

    h = (q**12 - 1) // r  # = 3^3 · l # where gcd(l, 3) = 1
    print(f"h is prime : {is_prime(h)}")
    assert math.gcd(r, h) == 1
    if curve_id.value == CurveID.BN254.value:
        base = 3
    elif curve_id.value == CurveID.BLS12_381.value:
        base = x - 1
        base = 3
    k, l = decompose_scalar_into_b_powers_and_remainder(h, base)
    assert base**k * l == h, f"{base}^k * l should be h"
    assert h % (base**k) == 0, f"h should be a multiple of {base}^k"
    assert math.gcd(l, base) == 1, f"l should be coprime with {base}"

    if curve_id.value == CurveID.BN254.value:
        λ = (
            6 * x + 2 + q - q**2 + q**3
        )  # https://eprint.iacr.org/2008/096.pdf See section 4 for BN curves.
    elif curve_id.value == CurveID.BLS12_381.value:
        λ = (
            -x + q
        )  # See https://gist.github.com/feltroidprime/6b43acc290fd6fd6d7f3cf2017047f62

    assert λ % r == 0, "λ should be a multiple of r. See section 4.2.2"
    m = λ // r
    d = math.gcd(m, h)

    # Theorem 3 proof fails: ?
    # assert (
    #     math.gcd(d**3, r * m) == 1
    # ), f"d should be coprime with r*m, but gcd is {math.gcd(d, r * m)}"
    print(f"d: {d}, (x-1): {x-1}")

    assert m % d == 0, "m should be a multiple of d"
    m_dash = m // d  # m' = m/d
    assert m_dash % h != 0
    f"m/d should not divide h. See section 4.2.2 Theorem 2."
    assert d * r * m_dash == λ, "incorrect parameters"  # sanity check
    assert math.gcd(λ, q**12 - 1) == d * r
    assert (
        math.gcd(m_dash, q**12 - 1) == 1
    ), f"m_dash should be coprime with q**12 - 1 'by construction'. See 4.3.2 computing m-th root"
    # equivalently, λ = 3rm′.
    # precompute r' and m''
    r_inv = pow(r, -1, h)
    m_d_inv = pow(m_dash, -1, h)
    return h, l, λ, m, m_dash, d, r, r_inv, m_d_inv


def be_to_int(bits: list[int]) -> int:
    """
    Convert a list of bits (1s and 0s), assuming big endian format, to an integer.

    Args:
    bits (list[int]): The list of bits (1s and 0s).

    Returns:
    int: The integer representation of the bits.
    """
    return int("".join(str(bit) for bit in bits), 2)


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


h_bn = (CURVES[BN254_ID].p ** 12 - 1) // CURVES[BN254_ID].n
h_bls = (CURVES[BLS12_381_ID].p ** 12 - 1) // CURVES[BLS12_381_ID].n
k_bn, l_bn = decompose_scalar_into_b_powers_and_remainder(h_bn, b=3)
k_bls, l_bls = decompose_scalar_into_b_powers_and_remainder(
    h_bls,
    b=(CURVES[BLS12_381_ID].x - 1),
)

print(f"k_bn: {k_bn}, l_bn: {l_bn}")
print(f"k_bls: {k_bls}, l_bls: {l_bls}")

E = be_to_int([0])

# Section 4.3.2 Finding c
# find some u a cubic non-residue and c such that f = c**λ * u.

# 1. Compute r-th root
# 2. Compute m′-th root
# 3. Compute cubic root


def pow_3_ord(a: E12):
    t = 0
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
    p = CURVES[f.curve_id].p
    exp = (p**12 - 1) // 3
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
    c = c**r_inv
    # 14 c ← c**m′′
    c = c**m_d_inv
    # 15 c ← c**1/3 (by using modified Tonelli-Shanks 4)
    c = find_cube_root(c, w, p)
    # 16 return (c, ws)
    return c, w**s


def find_c(f: Polynomial, w: E12) -> tuple[Polynomial, Polynomial]:
    assert f.degree() == 11, "incorrect degree of polynomial"
    f = int_to_e12(direct_to_tower(f.coefficients, BN254_ID, 12))
    c, wi = find_c_e12(f, w)
    return e12_to_direct_poly(c), e12_to_direct_poly(wi)


def print_e12(msg: str, a: E12):
    print(
        "\n",
        msg,
        "E12("
        f"""\n\t{a.c0.b0.a0},\n\t{a.c0.b0.a1},\n\t{a.c0.b1.a0},\n\t{a.c0.b1.a1},\n\t{a.c0.b2.a0},\n\t{a.c0.b2.a1},"""
        f"""\n\t{a.c1.b0.a0},\n\t{a.c1.b0.a1},\n\t{a.c1.b1.a0},\n\t{a.c1.b1.a1},\n\t{a.c1.b2.a0},\n\t{a.c1.b2.a1},"""
        "\n)\n",
    )


def print_poly(msg: str, a: Polynomial):
    print(
        "\n",
        msg,
        "Polynomial(",
        "\n\t" + ",\n\t".join([str(coeff.value) for coeff in a.coefficients]),
        "\n)\n",
    )


if __name__ == "__main__":
    f_coeffs = [
        0x1BF4E21820E6CC2B2DBC9453733A8D7C48F05E73F90ECC8BDD80505D2D3B1715,
        0x264F54F6B719920C4AC00AAFB3DF29CC8A9DDC25E264BDEE1ADE5E36077D58D7,
        0xDB269E3CD7ED27D825BCBAAEFB01023CF9B17BEED6092F7B96EAB87B571F3FE,
        0x25CE534442EE86A32C46B56D2BF289A0BE5F8703FB05C260B2CB820F2B253CF,
        0x33FC62C521F4FFDCB362B12220DB6C57F487906C0DAF4DC9BA736F882A420E1,
        0xE8B074995703E92A7B9568C90AE160E4D5B81AFFE628DC1D790241DE43D00D0,
        0x84E35BD0EEA3430B350041D235BB394E338E3A9ED2F0A9A1BA7FE786D391DE1,
        0x244D38253DA236F714CB763ABF68F7829EE631B4CC5EDE89B382E518D676D992,
        0x1EE0A098B62C76A9EBDF4D76C8DFC1586E3FCB6A01712CBDA8D10D07B32C5AF4,
        0xD23AEB23ACACF931F02ECA9ECEEE31EE9607EC003FF934694119A9C6CFFC4BD,
        0x16558217BB9B1BCDA995B123619808719CB8A282A190630E6D06D7D03E6333CA,
        0x14354C051802F8704939C9948EF91D89DB28FE9513AD7BBF58A4639AF347EA86,
    ]

    _ = build_params(CurveID.BN254)
    print(f"bn ok")
    h, l, λ, m, m_dash, d, r, r_inv, m_d_inv = build_params(CurveID.BLS12_381)

    h, l, λ, m, m_dash, d, r, r_inv, m_d_inv = build_params(CurveID.BN254)

    print("\n------------------ Testing with E12 ----------------------\n\n")
    FP12_ONE = E12.one(curve_id=BN254_ID)

    root_27th = int_to_e12(
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
        ]
    )

    assert root_27th**27 == FP12_ONE, "root_27th**27 should be one"
    assert root_27th**9 != FP12_ONE, "root_27th**9 should not be one"

    f = int_to_e12(f_coeffs)

    print("Computing residue witness for f,")
    print_e12("f =", f)
    print_e12("f**h =", f**h)
    c, wi = find_c_e12(f)
    c_inv = c.__inv__()

    print("residue witness c,")
    print_e12("c =", c)
    print_e12("c_inverse =", c_inv)
    print("witness scaling wi,")
    print_e12("wi = ", wi)

    result = c_inv**λ * f * wi
    print_e12("c_inv ** λ * f * wi (pairing) result:", result)
    assert result == FP12_ONE, "pairing not 1"

    print("\n--------------- Testing with Polynomial ------------------\n\n")

    f = int_tower_to_direct(f_coeffs)

    print("Computing residue witness for f,")
    print_poly("f =", f)

    c, wi = find_c(f)
    c_inv = c.inv(irreducible_poly)

    print("residue witness c,")
    print_poly("c =", c)
    print_poly("c_inverse =", c_inv)
    print("witness scaling wi,")
    print_poly("wi = ", wi)

    result: Polynomial = (c_inv.pow(λ, irreducible_poly) * f * wi) % irreducible_poly
    print_poly("c_inv ** λ * f * wi (pairing) result:", result)
    assert result.degree() == 0 and result.coefficients[0].value == 1, "pairing not 1"
