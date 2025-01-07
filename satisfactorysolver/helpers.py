# Copyright (c) 2024 Daniel Berlin and others
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import fractions
from decimal import Decimal
from fractions import Fraction


def rich_fraction_helper(value: Fraction) -> str:
    return format(value)


def rich_decimal_helper(value: Decimal) -> str:
    as_ratio = value.as_integer_ratio()
    if as_ratio[1] == 1 or as_ratio[1] % 10 == 0:
        return value.to_eng_string()
    else:
        return f"{as_ratio[0]}/{as_ratio[1]}"


def validate_fraction_helper(possible_fraction_str: str) -> Fraction | None:
    split_val = possible_fraction_str.split()
    if len(split_val) == 0:
        return None
    # Fraction class can parse individual fraction strings like 1/3, and whole numbers like 13,
    # but not mixed whole number + fraction like 13 1/3
    # This works by splitting the string into pieces, converting each into a fraction, then adding
    # the fractions.  So 13 1/3 is split to [13, 1/3], then converted to a list of fractions,
    # then the fractions are added together
    return sum(map(fractions.Fraction, split_val))


def validate_decimal_helper(possible_fraction_str: str) -> Decimal | None:
    possible_fraction = possible_fraction_str.split('/')
    if len(possible_fraction) == 1:
        return Decimal(possible_fraction[0])
    elif len(possible_fraction) == 2:
        return Decimal(int(possible_fraction[0])) / Decimal(int(possible_fraction[1]))
    else:
        return None
