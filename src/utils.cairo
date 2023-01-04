from starkware.cairo.common.uint256 import (
    Uint256,
    uint256_reverse_endian,
    uint256_unsigned_div_rem,
    uint256_mul,
    uint256_add,
    uint256_pow2,
)
from starkware.cairo.common.cairo_secp.bigint import (
    BASE,
    BigInt3,
    UnreducedBigInt3,
    UnreducedBigInt5,
    nondet_bigint3,
    bigint_mul,
)
from src.curve import P0, P1, P2
from starkware.cairo.common.cairo_builtins import BitwiseBuiltin
from starkware.cairo.common.registers import get_label_location
from starkware.cairo.common.math import unsigned_div_rem as felt_divmod, split_felt
from starkware.cairo.common.math_cmp import is_le
from starkware.cairo.common.alloc import alloc
from starkware.cairo.common.pow import pow

// returns 1 if x ==0 mod alt_bn128 prime
func is_zero{range_check_ptr}(x: BigInt3) -> (res: felt) {
    %{
        from starkware.cairo.common.cairo_secp.secp_utils import pack
        P = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
        x = pack(ids.x, PRIME) % P
    %}
    if (nondet %{ x == 0 %} != 0) {
        verify_zero3(x);
        // verify_zero5(UnreducedBigInt5(d0=x.d0, d1=x.d1, d2=x.d2, d3=0, d4=0))
        return (res=1);
    }

    %{
        from starkware.python.math_utils import div_mod
        value = x_inv = div_mod(1, x, P)
    %}
    let (x_inv) = nondet_bigint3();
    let (x_x_inv) = bigint_mul(x, x_inv);

    // Check that x * x_inv = 1 to verify that x != 0.
    verify_zero5(
        UnreducedBigInt5(
            d0=x_x_inv.d0 - 1, d1=x_x_inv.d1, d2=x_x_inv.d2, d3=x_x_inv.d3, d4=x_x_inv.d4
        ),
    );
    return (res=0);
}

// y MUST be a power of 2
func bitwise_divmod{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}(x: felt, y: felt) -> (
    x_and_y: felt, r: felt
) {
    assert bitwise_ptr.x = x;
    assert bitwise_ptr.y = y - 1;
    let x_and_y = bitwise_ptr.x_and_y;

    let bitwise_ptr = bitwise_ptr + BitwiseBuiltin.SIZE;
    return (x_and_y=(x - x_and_y) / y, r=x_and_y);
}
func felt_divmod_no_input_check{range_check_ptr}(value, div) -> (q: felt, r: felt) {
    // let r = [range_check_ptr];
    // let q = [range_check_ptr + 1];
    // let range_check_ptr = range_check_ptr + 2;
    alloc_locals;
    local r;
    local q;
    %{
        from starkware.cairo.common.math_utils import assert_integer
        assert_integer(ids.div)
        assert 0 < ids.div <= PRIME // range_check_builtin.bound, \
            f'div={hex(ids.div)} is out of the valid range.'
        ids.q, ids.r = divmod(ids.value, ids.div)
    %}

    assert [range_check_ptr] = div - 1 - r;
    let range_check_ptr = range_check_ptr + 1;
    // assert_le(r, div - 1);

    assert value = q * div + r;
    return (q, r);
}

func verify_zero3{range_check_ptr}(val: BigInt3) {
    alloc_locals;
    local flag;
    local q;
    %{
        from starkware.cairo.common.cairo_secp.secp_utils import pack
        from starkware.cairo.common.math_utils import as_int

        P = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47

        v = pack(ids.val, PRIME) 
        q, r = divmod(v, P)
        assert r == 0, f"verify_zero: Invalid input {ids.val.d0, ids.val.d1, ids.val.d2}."

        ids.flag = 1 if q > 0 else 0
        q = q if q > 0 else 0-q
        ids.q = q % PRIME
    %}
    assert [range_check_ptr] = q + 2 ** 127;

    tempvar carry1 = ((2 * flag - 1) * q * P0 - val.d0) / BASE;
    assert [range_check_ptr + 1] = carry1 + 2 ** 127;

    tempvar carry2 = ((2 * flag - 1) * q * P1 - val.d1 + carry1) / BASE;
    assert [range_check_ptr + 2] = carry2 + 2 ** 127;

    assert (2 * flag - 1) * q * P2 - val.d2 + carry2 = 0;

    let range_check_ptr = range_check_ptr + 3;

    return ();
}

func verify_zero5{range_check_ptr}(val: UnreducedBigInt5) {
    alloc_locals;
    local flag;
    local q1;
    %{
        from starkware.cairo.common.cairo_secp.secp_utils import pack
        from starkware.cairo.common.math_utils import as_int

        P = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47

        v3 = as_int(ids.val.d3, PRIME)
        v4 = as_int(ids.val.d4, PRIME)
        v = pack(ids.val, PRIME) + v3*2**258 + v4*2**344

        q, r = divmod(v, P)
        assert r == 0, f"verify_zero: Invalid input {ids.val.d0, ids.val.d1, ids.val.d2, ids.val.d3, ids.val.d4}."

        # Since q usually doesn't fit BigInt3, divide it again
        ids.flag = 1 if q > 0 else 0
        q = q if q > 0 else 0-q
        q1, q2 = divmod(q, P)
        ids.q1 = q1
        value = k = q2
    %}
    let (k) = nondet_bigint3();
    let fullk = BigInt3(q1 * P0 + k.d0, q1 * P1 + k.d1, q1 * P2 + k.d2);
    let P = BigInt3(P0, P1, P2);
    let (k_n) = bigint_mul(fullk, P);

    // val mod n = 0, so val = k_n
    tempvar carry1 = ((2 * flag - 1) * k_n.d0 - val.d0) / BASE;
    assert [range_check_ptr + 0] = carry1 + 2 ** 127;

    tempvar carry2 = ((2 * flag - 1) * k_n.d1 - val.d1 + carry1) / BASE;
    assert [range_check_ptr + 1] = carry2 + 2 ** 127;

    tempvar carry3 = ((2 * flag - 1) * k_n.d2 - val.d2 + carry2) / BASE;
    assert [range_check_ptr + 2] = carry3 + 2 ** 127;

    tempvar carry4 = ((2 * flag - 1) * k_n.d3 - val.d3 + carry3) / BASE;
    assert [range_check_ptr + 3] = carry4 + 2 ** 127;

    assert (2 * flag - 1) * k_n.d4 - val.d4 + carry4 = 0;

    let range_check_ptr = range_check_ptr + 4;

    return ();
}
func get_felt_bitlength{range_check_ptr, bitwise_ptr: BitwiseBuiltin*}(x: felt) -> felt {
    alloc_locals;
    local bit_length;
    %{
        x = ids.x
        ids.bit_length = x.bit_length()
    %}

    // Next two lines Not necessary : will fail if pow2(bit_length) is too big, unknown cell.
    // let le = is_le(bit_length, 252);
    // assert le = 1;
    assert bitwise_ptr[0].x = x;
    let n = pow2(bit_length);
    assert bitwise_ptr[0].y = n - 1;
    tempvar word = bitwise_ptr[0].x_and_y;
    assert word = x;

    assert bitwise_ptr[1].x = x;

    let n = pow2(bit_length - 1);

    assert bitwise_ptr[1].y = n - 1;
    tempvar word = bitwise_ptr[1].x_and_y;
    assert word = x - n;

    let bitwise_ptr = bitwise_ptr + 2 * BitwiseBuiltin.SIZE;
    return bit_length;
}

func pow2(i) -> felt {
    let (data_address) = get_label_location(data);
    return [data_address + i];

    data:
    dw 0x1;
    dw 0x2;
    dw 0x4;
    dw 0x8;
    dw 0x10;
    dw 0x20;
    dw 0x40;
    dw 0x80;
    dw 0x100;
    dw 0x200;
    dw 0x400;
    dw 0x800;
    dw 0x1000;
    dw 0x2000;
    dw 0x4000;
    dw 0x8000;
    dw 0x10000;
    dw 0x20000;
    dw 0x40000;
    dw 0x80000;
    dw 0x100000;
    dw 0x200000;
    dw 0x400000;
    dw 0x800000;
    dw 0x1000000;
    dw 0x2000000;
    dw 0x4000000;
    dw 0x8000000;
    dw 0x10000000;
    dw 0x20000000;
    dw 0x40000000;
    dw 0x80000000;
    dw 0x100000000;
    dw 0x200000000;
    dw 0x400000000;
    dw 0x800000000;
    dw 0x1000000000;
    dw 0x2000000000;
    dw 0x4000000000;
    dw 0x8000000000;
    dw 0x10000000000;
    dw 0x20000000000;
    dw 0x40000000000;
    dw 0x80000000000;
    dw 0x100000000000;
    dw 0x200000000000;
    dw 0x400000000000;
    dw 0x800000000000;
    dw 0x1000000000000;
    dw 0x2000000000000;
    dw 0x4000000000000;
    dw 0x8000000000000;
    dw 0x10000000000000;
    dw 0x20000000000000;
    dw 0x40000000000000;
    dw 0x80000000000000;
    dw 0x100000000000000;
    dw 0x200000000000000;
    dw 0x400000000000000;
    dw 0x800000000000000;
    dw 0x1000000000000000;
    dw 0x2000000000000000;
    dw 0x4000000000000000;
    dw 0x8000000000000000;
    dw 0x10000000000000000;
    dw 0x20000000000000000;
    dw 0x40000000000000000;
    dw 0x80000000000000000;
    dw 0x100000000000000000;
    dw 0x200000000000000000;
    dw 0x400000000000000000;
    dw 0x800000000000000000;
    dw 0x1000000000000000000;
    dw 0x2000000000000000000;
    dw 0x4000000000000000000;
    dw 0x8000000000000000000;
    dw 0x10000000000000000000;
    dw 0x20000000000000000000;
    dw 0x40000000000000000000;
    dw 0x80000000000000000000;
    dw 0x100000000000000000000;
    dw 0x200000000000000000000;
    dw 0x400000000000000000000;
    dw 0x800000000000000000000;
    dw 0x1000000000000000000000;
    dw 0x2000000000000000000000;
    dw 0x4000000000000000000000;
    dw 0x8000000000000000000000;
    dw 0x10000000000000000000000;
    dw 0x20000000000000000000000;
    dw 0x40000000000000000000000;
    dw 0x80000000000000000000000;
    dw 0x100000000000000000000000;
    dw 0x200000000000000000000000;
    dw 0x400000000000000000000000;
    dw 0x800000000000000000000000;
    dw 0x1000000000000000000000000;
    dw 0x2000000000000000000000000;
    dw 0x4000000000000000000000000;
    dw 0x8000000000000000000000000;
    dw 0x10000000000000000000000000;
    dw 0x20000000000000000000000000;
    dw 0x40000000000000000000000000;
    dw 0x80000000000000000000000000;
    dw 0x100000000000000000000000000;
    dw 0x200000000000000000000000000;
    dw 0x400000000000000000000000000;
    dw 0x800000000000000000000000000;
    dw 0x1000000000000000000000000000;
    dw 0x2000000000000000000000000000;
    dw 0x4000000000000000000000000000;
    dw 0x8000000000000000000000000000;
    dw 0x10000000000000000000000000000;
    dw 0x20000000000000000000000000000;
    dw 0x40000000000000000000000000000;
    dw 0x80000000000000000000000000000;
    dw 0x100000000000000000000000000000;
    dw 0x200000000000000000000000000000;
    dw 0x400000000000000000000000000000;
    dw 0x800000000000000000000000000000;
    dw 0x1000000000000000000000000000000;
    dw 0x2000000000000000000000000000000;
    dw 0x4000000000000000000000000000000;
    dw 0x8000000000000000000000000000000;
    dw 0x10000000000000000000000000000000;
    dw 0x20000000000000000000000000000000;
    dw 0x40000000000000000000000000000000;
    dw 0x80000000000000000000000000000000;
    dw 0x100000000000000000000000000000000;
    dw 0x200000000000000000000000000000000;
    dw 0x400000000000000000000000000000000;
    dw 0x800000000000000000000000000000000;
    dw 0x1000000000000000000000000000000000;
    dw 0x2000000000000000000000000000000000;
    dw 0x4000000000000000000000000000000000;
    dw 0x8000000000000000000000000000000000;
    dw 0x10000000000000000000000000000000000;
    dw 0x20000000000000000000000000000000000;
    dw 0x40000000000000000000000000000000000;
    dw 0x80000000000000000000000000000000000;
    dw 0x100000000000000000000000000000000000;
    dw 0x200000000000000000000000000000000000;
    dw 0x400000000000000000000000000000000000;
    dw 0x800000000000000000000000000000000000;
    dw 0x1000000000000000000000000000000000000;
    dw 0x2000000000000000000000000000000000000;
    dw 0x4000000000000000000000000000000000000;
    dw 0x8000000000000000000000000000000000000;
    dw 0x10000000000000000000000000000000000000;
    dw 0x20000000000000000000000000000000000000;
    dw 0x40000000000000000000000000000000000000;
    dw 0x80000000000000000000000000000000000000;
    dw 0x100000000000000000000000000000000000000;
    dw 0x200000000000000000000000000000000000000;
    dw 0x400000000000000000000000000000000000000;
    dw 0x800000000000000000000000000000000000000;
    dw 0x1000000000000000000000000000000000000000;
    dw 0x2000000000000000000000000000000000000000;
    dw 0x4000000000000000000000000000000000000000;
    dw 0x8000000000000000000000000000000000000000;
    dw 0x10000000000000000000000000000000000000000;
    dw 0x20000000000000000000000000000000000000000;
    dw 0x40000000000000000000000000000000000000000;
    dw 0x80000000000000000000000000000000000000000;
    dw 0x100000000000000000000000000000000000000000;
    dw 0x200000000000000000000000000000000000000000;
    dw 0x400000000000000000000000000000000000000000;
    dw 0x800000000000000000000000000000000000000000;
    dw 0x1000000000000000000000000000000000000000000;
    dw 0x2000000000000000000000000000000000000000000;
    dw 0x4000000000000000000000000000000000000000000;
    dw 0x8000000000000000000000000000000000000000000;
    dw 0x10000000000000000000000000000000000000000000;
    dw 0x20000000000000000000000000000000000000000000;
    dw 0x40000000000000000000000000000000000000000000;
    dw 0x80000000000000000000000000000000000000000000;
    dw 0x100000000000000000000000000000000000000000000;
    dw 0x200000000000000000000000000000000000000000000;
    dw 0x400000000000000000000000000000000000000000000;
    dw 0x800000000000000000000000000000000000000000000;
    dw 0x1000000000000000000000000000000000000000000000;
    dw 0x2000000000000000000000000000000000000000000000;
    dw 0x4000000000000000000000000000000000000000000000;
    dw 0x8000000000000000000000000000000000000000000000;
    dw 0x10000000000000000000000000000000000000000000000;
    dw 0x20000000000000000000000000000000000000000000000;
    dw 0x40000000000000000000000000000000000000000000000;
    dw 0x80000000000000000000000000000000000000000000000;
    dw 0x100000000000000000000000000000000000000000000000;
    dw 0x200000000000000000000000000000000000000000000000;
    dw 0x400000000000000000000000000000000000000000000000;
    dw 0x800000000000000000000000000000000000000000000000;
    dw 0x1000000000000000000000000000000000000000000000000;
    dw 0x2000000000000000000000000000000000000000000000000;
    dw 0x4000000000000000000000000000000000000000000000000;
    dw 0x8000000000000000000000000000000000000000000000000;
    dw 0x10000000000000000000000000000000000000000000000000;
    dw 0x20000000000000000000000000000000000000000000000000;
    dw 0x40000000000000000000000000000000000000000000000000;
    dw 0x80000000000000000000000000000000000000000000000000;
    dw 0x100000000000000000000000000000000000000000000000000;
    dw 0x200000000000000000000000000000000000000000000000000;
    dw 0x400000000000000000000000000000000000000000000000000;
    dw 0x800000000000000000000000000000000000000000000000000;
    dw 0x1000000000000000000000000000000000000000000000000000;
    dw 0x2000000000000000000000000000000000000000000000000000;
    dw 0x4000000000000000000000000000000000000000000000000000;
    dw 0x8000000000000000000000000000000000000000000000000000;
    dw 0x10000000000000000000000000000000000000000000000000000;
    dw 0x20000000000000000000000000000000000000000000000000000;
    dw 0x40000000000000000000000000000000000000000000000000000;
    dw 0x80000000000000000000000000000000000000000000000000000;
    dw 0x100000000000000000000000000000000000000000000000000000;
    dw 0x200000000000000000000000000000000000000000000000000000;
    dw 0x400000000000000000000000000000000000000000000000000000;
    dw 0x800000000000000000000000000000000000000000000000000000;
    dw 0x1000000000000000000000000000000000000000000000000000000;
    dw 0x2000000000000000000000000000000000000000000000000000000;
    dw 0x4000000000000000000000000000000000000000000000000000000;
    dw 0x8000000000000000000000000000000000000000000000000000000;
    dw 0x10000000000000000000000000000000000000000000000000000000;
    dw 0x20000000000000000000000000000000000000000000000000000000;
    dw 0x40000000000000000000000000000000000000000000000000000000;
    dw 0x80000000000000000000000000000000000000000000000000000000;
    dw 0x100000000000000000000000000000000000000000000000000000000;
    dw 0x200000000000000000000000000000000000000000000000000000000;
    dw 0x400000000000000000000000000000000000000000000000000000000;
    dw 0x800000000000000000000000000000000000000000000000000000000;
    dw 0x1000000000000000000000000000000000000000000000000000000000;
    dw 0x2000000000000000000000000000000000000000000000000000000000;
    dw 0x4000000000000000000000000000000000000000000000000000000000;
    dw 0x8000000000000000000000000000000000000000000000000000000000;
    dw 0x10000000000000000000000000000000000000000000000000000000000;
    dw 0x20000000000000000000000000000000000000000000000000000000000;
    dw 0x40000000000000000000000000000000000000000000000000000000000;
    dw 0x80000000000000000000000000000000000000000000000000000000000;
    dw 0x100000000000000000000000000000000000000000000000000000000000;
    dw 0x200000000000000000000000000000000000000000000000000000000000;
    dw 0x400000000000000000000000000000000000000000000000000000000000;
    dw 0x800000000000000000000000000000000000000000000000000000000000;
    dw 0x1000000000000000000000000000000000000000000000000000000000000;
    dw 0x2000000000000000000000000000000000000000000000000000000000000;
    dw 0x4000000000000000000000000000000000000000000000000000000000000;
    dw 0x8000000000000000000000000000000000000000000000000000000000000;
    dw 0x10000000000000000000000000000000000000000000000000000000000000;
    dw 0x20000000000000000000000000000000000000000000000000000000000000;
    dw 0x40000000000000000000000000000000000000000000000000000000000000;
    dw 0x80000000000000000000000000000000000000000000000000000000000000;
    dw 0x100000000000000000000000000000000000000000000000000000000000000;
    dw 0x200000000000000000000000000000000000000000000000000000000000000;
    dw 0x400000000000000000000000000000000000000000000000000000000000000;
}
