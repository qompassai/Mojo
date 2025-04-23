# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Provides the `hex` and `bin` functions.

These are Mojo built-ins, so you don't need to import them.
"""

from collections import Optional
from utils import InlineArray

alias _DEFAULT_DIGIT_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


# ===----------------------------------------------------------------------===#
# bin
# ===----------------------------------------------------------------------===#


@always_inline
fn bin[
    type: DType
](num: Scalar[type], prefix: StringLiteral = "0b", /) -> String:
    """Return the binary string representation an integral value.

    ```mojo
    print(bin(123))
    print(bin(-123))
    ```
    ```plaintext
    '0b1111011'
    '-0b1111011'
    ```

    Parameters:
        type: The data type of the integral scalar.

    Args:
        num: An integral scalar value.
        prefix: The prefix of the formatted int.

    Returns:
        The binary string representation of num.
    """
    return _try_format_int(num, 2, prefix=prefix)


# Need this until we have constraints to stop the compiler from matching this
# directly to bin[type: DType](num: Scalar[type]).
@always_inline("nodebug")
fn bin(b: Scalar[DType.bool], prefix: StringLiteral = "0b", /) -> String:
    """Returns the binary representation of a scalar bool.

    Args:
        b: A scalar bool value.
        prefix: The prefix of the formatted int.

    Returns:
        The binary string representation of b.
    """
    return bin(b.cast[DType.int8]())


@always_inline("nodebug")
fn bin[T: Indexer](num: T, prefix: StringLiteral = "0b", /) -> String:
    """Returns the binary representation of an indexer type.

    Parameters:
        T: The Indexer type.

    Args:
        num: An indexer value.
        prefix: The prefix of the formatted int.

    Returns:
        The binary string representation of num.
    """
    return bin(Scalar[DType.index](index(num)))


# ===----------------------------------------------------------------------===#
# hex
# ===----------------------------------------------------------------------===#


@always_inline
fn hex[
    type: DType
](value: Scalar[type], prefix: StringLiteral = "0x", /) -> String:
    """Returns the hex string representation of the given integer.

    The hexadecimal representation is a base-16 encoding of the integer value.

    The returned string will be prefixed with "0x" to indicate that the
    subsequent digits are hex.

    Parameters:
        type: The type of the Scalar to represent in hexadecimal.

    Args:
        value: The integer value to format.
        prefix: The prefix of the formatted int.

    Returns:
        A string containing the hex representation of the given integer.
    """
    return _try_format_int(value, 16, prefix=prefix)


@always_inline
fn hex[T: Indexer](value: T, prefix: StringLiteral = "0x", /) -> String:
    """Returns the hex string representation of the given integer.

    The hexadecimal representation is a base-16 encoding of the integer value.

    The returned string will be prefixed with "0x" to indicate that the
    subsequent digits are hex.

    Parameters:
        T: The indexer type to represent in hexadecimal.

    Args:
        value: The integer value to format.
        prefix: The prefix of the formatted int.

    Returns:
        A string containing the hex representation of the given integer.
    """
    return hex[DType.index](index(value), prefix)


@always_inline
fn hex(value: Scalar[DType.bool], prefix: StringLiteral = "0x", /) -> String:
    """Returns the hex string representation of the given scalar bool.

    The hexadecimal representation is a base-16 encoding of the bool.

    The returned string will be prefixed with "0x" to indicate that the
    subsequent digits are hex.

    Args:
        value: The bool value to format.
        prefix: The prefix of the formatted int.

    Returns:
        A string containing the hex representation of the given bool.
    """
    return hex(value.cast[DType.int8]())


# ===----------------------------------------------------------------------===#
# Integer formatting utilities
# ===----------------------------------------------------------------------===#


fn _try_format_int[
    type: DType
](value: Scalar[type], radix: Int = 10, prefix: StringLiteral = "",) -> String:
    try:
        return _format_int(value, radix, prefix=prefix)
    except e:
        # This should not be reachable as _format_int only throws if we pass
        # incompatible radix and custom digit chars, which we aren't doing
        # above.
        return abort[String](
            "unexpected exception formatting value as hexadecimal: " + str(e)
        )


fn _format_int[
    type: DType
](
    value: Scalar[type],
    radix: Int = 10,
    digit_chars: StringLiteral = _DEFAULT_DIGIT_CHARS,
    prefix: StringLiteral = "",
) raises -> String:
    var string = String()
    var fmt = string._unsafe_to_formatter()

    _write_int(fmt, value, radix, digit_chars, prefix)

    return string^


@always_inline
fn _write_int[
    type: DType
](
    inout fmt: Formatter,
    value: Scalar[type],
    radix: Int = 10,
    digit_chars: StringLiteral = _DEFAULT_DIGIT_CHARS,
    prefix: StringLiteral = "",
) raises:
    var err = _try_write_int(fmt, value, radix, digit_chars, prefix)
    if err:
        raise err.value()[]


@always_inline
fn _try_write_int[
    type: DType
](
    inout fmt: Formatter,
    value: Scalar[type],
    radix: Int = 10,
    digit_chars: StringLiteral = _DEFAULT_DIGIT_CHARS,
    prefix: StringLiteral = "",
) -> Optional[Error]:
    """Writes a formatted string representation of the given integer using the specified radix.

    The maximum supported radix is 36 unless a custom `digit_chars` mapping is
    provided.
    """

    #
    # Check that the radix and available digit characters are valid
    #

    constrained[type.is_integral(), "Expected integral"]()

    if radix < 2:
        return Error("Unable to format integer to string with radix < 2")

    if radix > len(digit_chars):
        return Error(
            "Unable to format integer to string when provided radix is larger "
            "than length of available digit value characters"
        )

    if not len(digit_chars) >= 2:
        return Error(
            "Unable to format integer to string when provided digit_chars"
            " mapping len is not >= 2"
        )

    #
    # Process the integer value into its corresponding digits
    #

    # TODO(#26444, Unicode support): Get an array of Character, not bytes.
    var digit_chars_array = digit_chars.unsafe_ptr()

    # Prefix a '-' if the original int was negative and make positive.
    if value < 0:
        fmt.write_str("-")

    # Add the custom number prefix, e.g. "0x" commonly used for hex numbers.
    # This comes *after* the minus sign, if present.
    fmt.write_str(prefix)

    if value == 0:
        # TODO: Replace with safe digit_chars[:1] syntax.
        # SAFETY:
        #   This static lifetime is valid as long as we're using a
        #   `StringLiteral` for `digit_chars`.
        var zero = StringSlice[False, ImmutableStaticLifetime](
            # TODO: Remove cast after transition to UInt8 strings is complete.
            unsafe_from_utf8_ptr=digit_chars_array.bitcast[UInt8](),
            len=1,
        )
        fmt.write_str(zero)
        return

    #
    # Create a buffer to store the formatted value
    #

    # Stack allocate enough bytes to store any formatted 64-bit integer
    # TODO: use a dynamic size when #2194 is resolved
    alias CAPACITY: Int = 64

    var buf = InlineArray[Int8, CAPACITY](unsafe_uninitialized=True)

    # Start the buf pointer at the end. We will write the least-significant
    # digits later in the buffer, and then decrement the pointer to move
    # earlier in the buffer as we write the more-significant digits.
    var offset = CAPACITY - 1

    #
    # Write the digits of the number
    #

    var remaining_int = value

    @parameter
    fn process_digits[get_digit_value: fn () capturing -> Scalar[type]]():
        while remaining_int:
            var digit_value = get_digit_value()

            # Write the char representing the value of the least significant
            # digit.
            buf[offset] = digit_chars_array[int(digit_value)]

            # Position the offset to write the next digit.
            offset -= 1

            # Drop the least significant digit
            remaining_int /= radix

    if remaining_int >= 0:

        @parameter
        fn pos_digit_value() -> Scalar[type]:
            return remaining_int % radix

        process_digits[pos_digit_value]()
    else:

        @parameter
        fn neg_digit_value() -> Scalar[type]:
            return abs(remaining_int % -radix)

        process_digits[neg_digit_value]()

    # Re-add +1 byte since the loop ended so we didn't write another char.
    offset += 1

    var buf_ptr = buf.unsafe_ptr() + offset

    # Calculate the length of the buffer we've filled. This is the number of
    # bytes from our final `buf_ptr` to the end of the buffer.
    var len = CAPACITY - offset

    # SAFETY:
    #   Create a slice to only those bytes in `buf` that have been initialized.
    var str_slice = StringSlice[False, __lifetime_of(buf)](
        # TODO: Remove cast after transition to UInt8 strings is complete.
        unsafe_from_utf8_ptr=buf_ptr.bitcast[UInt8](),
        len=len,
    )

    fmt.write_str(str_slice)

    return None
