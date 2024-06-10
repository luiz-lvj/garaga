import math
from dataclasses import dataclass
import logging


@dataclass(slots=True)
class Curve:
    name: str
    q: int
    x: int
    r: int


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Remove all handlers associated with the logger object.
for handler in logger.handlers[:]:
    logger.removeHandler(handler)


# Create a handler that will print the log messages in red
class RedWarningHandler(logging.StreamHandler):
    def emit(self, record):
        # Use ANSI escape codes to set the text color to red
        red_code = "\033[91m"
        reset_code = "\033[0m"
        # Format message to be clickable in VS Code
        fs = f"{self.formatter._fmt}"
        record.msg = (
            f"{red_code}{record.pathname}:{record.lineno}: {record.msg}{reset_code}"
        )
        super().emit(record)


# Define a formatter that includes the file name and line number
formatter = logging.Formatter("%(filename)s:%(lineno)d: %(message)s")

# Set the formatter for the handler
handler = RedWarningHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def build_params(curve: Curve) -> tuple[int, int, int, int, int, int]:
    q = curve.q
    x = curve.x
    r = curve.r

    h = (q**12 - 1) // r

    if math.gcd(r, h) != 1:
        logger.warning(f"{curve.name}: gcd(r, h) should be 1")

    if curve.name == "BN":
        base = 3
    elif curve.name == "BLS":
        base = 3  # Is coprime with
        base = x - 1

    k, l = decompose_scalar_into_b_powers_and_remainder(h, base)

    if base**k * l != h:
        logger.warning(f"{curve.name}: {base}^k * l should be h")
    if h % (base**k) != 0:
        logger.warning(f"{curve.name}: h should be a multiple of {base}^k")
    if math.gcd(l, base) != 1:
        logger.warning(f"{curve.name}: l should be coprime with {base}")

    if curve.name == "BN":
        λ = 6 * x + 2 + q - q**2 + q**3
    elif curve.name == "BLS":
        λ = -x + q

    if λ % r != 0:
        logger.warning(f"{curve.name}: λ should be a multiple of r. See section 4.2.2")

    m = λ // r
    d = math.gcd(m, h)
    m_dash = m // d

    print(f"{curve.name}: d: {d}, (x-1): {x-1}")

    if m % d != 0:
        logger.warning(f"{curve.name}: m should be a multiple of d")
    if m_dash % h == 0:
        logger.warning(
            f"{curve.name}: m/d should not divide h. See section 4.2.2 Theorem 2."
        )
    if d * r * m_dash != λ:
        logger.warning(f"{curve.name}: incorrect parameters")
    if math.gcd(λ, q**12 - 1) != d * r:
        logger.warning(f"{curve.name}: gcd(λ, q**12 - 1) should be d * r")
    if math.gcd(m_dash, q**12 - 1) != 1:
        logger.warning(
            f"{curve.name}: m_dash should be coprime with q**12 - 1 'by construction'. See 4.3.2 computing m-th root"
        )

    if math.gcd(d, r * m) != 1:
        logger.warning(
            f"{curve.name}: d should be coprime with r*m, but gcd is {math.gcd(d, r * m)}. See theorem 3 proof."
        )

    r_inv = pow(r, -1, h)
    m_d_inv = pow(m_dash, -1, h)
    return h, l, λ, m, m_dash, d, r, r_inv, m_d_inv


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


if __name__ == "__main__":
    bn = Curve(
        name="BN",
        q=0x30644E72E131A029B85045B68181585D97816A916871CA8D3C208C16D87CFD47,
        x=0x44E992B44A6909F1,
        r=0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001,
    )

    bls = Curve(
        name="BLS",
        q=0x1A0111EA397FE69A4B1BA7B6434BACD764774B84F38512BF6730D2A0F6B0F6241EABFFFEB153FFFFB9FEFFFFFFFFAAAB,
        x=-0xD201000000010000,
        r=0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001,
    )

    build_params(bn)
    build_params(bls)
