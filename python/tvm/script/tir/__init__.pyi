# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=redefined-builtin
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Union,
    Sequence,
    List,
    Mapping,
    overload,
)
from numbers import Number
import builtins

from tvm.tir.function import PrimFunc
from tvm.tir import Range
from tvm.runtime import Object
from .node import BufferSlice

"""
redefine types
"""

class PrimExpr:
    def __init__(self: PrimExpr) -> None: ...
    @overload
    def __add__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __add__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __sub__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __sub__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __mul__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __mul__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    @overload
    def __div__(self: PrimExpr, other: PrimExpr) -> PrimExpr: ...
    @overload
    def __div__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    def __radd__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    def __rsub__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    def __rmul__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...
    def __rdiv__(self: PrimExpr, other: Union[int, float]) -> PrimExpr: ...

class Var(PrimExpr): ...
class IterVar(Var): ...

class Buffer:
    @overload
    def __getitem__(self: Buffer, pos: Sequence[Union[PrimExpr, int, slice]]) -> PrimExpr: ...
    @overload
    def __getitem__(self: Buffer, pos: Union[PrimExpr, int, slice]) -> PrimExpr: ...
    @overload
    def __setitem__(
        self: Buffer, pos: Sequence[Union[PrimExpr, int, slice]], value: PrimExpr
    ) -> None: ...
    @overload
    def __setitem__(self: Buffer, pos: Union[PrimExpr, int, slice], value: PrimExpr) -> None: ...
    @property
    def data(self: Buffer) -> Ptr: ...

"""
Variables and constants
"""

def bool(imm: Union[PrimExpr, builtins.bool, builtins.int]) -> PrimExpr: ...
def int8(imm: Union[PrimExpr, int]) -> PrimExpr: ...
def int16(imm: Union[PrimExpr, int]) -> PrimExpr: ...
def int32(imm: Union[PrimExpr, int]) -> PrimExpr: ...
def int64(imm: Union[PrimExpr, int]) -> PrimExpr: ...
def uint8(imm: Union[PrimExpr, int]) -> PrimExpr: ...
def uint16(imm: Union[PrimExpr, int]) -> PrimExpr: ...
def uint32(imm: Union[PrimExpr, int]) -> PrimExpr: ...
def uint64(imm: Union[PrimExpr, int]) -> PrimExpr: ...
def float8(imm: Union[PrimExpr, int]) -> PrimExpr: ...
def float16(imm: Union[PrimExpr, int]) -> PrimExpr: ...
def float32(imm: Union[PrimExpr, int]) -> PrimExpr: ...
def float64(imm: Union[PrimExpr, int]) -> PrimExpr: ...

"""
Intrinsic
"""

def min_value(dtype: str) -> PrimExpr: ...
def max_value(dtype: str) -> PrimExpr: ...
def floordiv(x: PrimExpr, y: PrimExpr) -> PrimExpr: ...
def floormod(x: PrimExpr, y: PrimExpr) -> PrimExpr: ...
def truncmod(x: PrimExpr, y: PrimExpr) -> PrimExpr: ...
def truncdiv(x: PrimExpr, y: PrimExpr) -> PrimExpr: ...
def abs(x: PrimExpr) -> PrimExpr: ...
def load(
    dtype: str, var: Var, index: PrimExpr, predicate: Union[PrimExpr, builtins.bool] = None
) -> PrimExpr: ...
def cast(value: PrimExpr, dtype: str) -> PrimExpr: ...
def ramp(base: PrimExpr, stride: Any, lanes: int) -> PrimExpr: ...
def broadcast(value: PrimExpr, lanes: int) -> PrimExpr: ...
def iter_var(var: Union[Var, str], dom: Range, iter_type: int, thread_tag: str) -> IterVar: ...
def max(a: PrimExpr, b: PrimExpr) -> PrimExpr: ...
def min(a: PrimExpr, b: PrimExpr) -> PrimExpr: ...
def Select(cond: PrimExpr, if_body: PrimExpr, else_body: PrimExpr) -> PrimExpr: ...
def if_then_else(cond: PrimExpr, t: PrimExpr, f: PrimExpr, dtype: str) -> PrimExpr: ...
def evaluate(value: PrimExpr) -> None: ...
def reinterpret(value: PrimExpr, dtype: str) -> PrimExpr: ...
def store(
    var: Var, index: PrimExpr, value: PrimExpr, predicate: Union[PrimExpr, builtins.bool] = True
) -> None: ...
def comm_reducer(lambda_io: Callable[[Any, Any], Any], identities: List[PrimExpr]) -> PrimExpr: ...

"""
Intrinsics - tvm builtin 
"""

def tvm_thread_allreduce(
    *freduceargs: Union[PrimExpr, builtins.bool, Ptr], dtype: str
) -> PrimExpr: ...

"""
Unary operator
Note that any intrinsics not registered in script.tir.intrin
should add "dtype" as an argument. This is different from their
definition but intentional.
"""

def exp(x: PrimExpr, dtype: str) -> PrimExpr: ...
def exp2(x: PrimExpr, dtype: str) -> PrimExpr: ...
def exp10(x: PrimExpr, dtype: str) -> PrimExpr: ...
def erf(x: PrimExpr, dtype: str) -> PrimExpr: ...
def tanh(x: PrimExpr, dtype: str) -> PrimExpr: ...
def sigmoid(x: PrimExpr, dtype: str) -> PrimExpr: ...
def log(x: PrimExpr, dtype: str) -> PrimExpr: ...
def log2(x: PrimExpr, dtype: str) -> PrimExpr: ...
def log10(x: PrimExpr, dtype: str) -> PrimExpr: ...
def log1p(x: PrimExpr, dtype: str) -> PrimExpr: ...
def tan(x: PrimExpr, dtype: str) -> PrimExpr: ...
def cos(x: PrimExpr, dtype: str) -> PrimExpr: ...
def cosh(x: PrimExpr, dtype: str) -> PrimExpr: ...
def acos(x: PrimExpr, dtype: str) -> PrimExpr: ...
def acosh(x: PrimExpr, dtype: str) -> PrimExpr: ...
def sin(x: PrimExpr, dtype: str) -> PrimExpr: ...
def sinh(x: PrimExpr, dtype: str) -> PrimExpr: ...
def asin(x: PrimExpr, dtype: str) -> PrimExpr: ...
def asinh(x: PrimExpr, dtype: str) -> PrimExpr: ...
def atan(x: PrimExpr, dtype: str) -> PrimExpr: ...
def atanh(x: PrimExpr, dtype: str) -> PrimExpr: ...
def atan2(x: PrimExpr, dtype: str) -> PrimExpr: ...
def sqrt(x: PrimExpr, dtype: str) -> PrimExpr: ...
def rsqrt(x: PrimExpr, dtype: str) -> PrimExpr: ...

"""
special_stmt - Buffers
"""

def match_buffer(
    param: Union[Var, BufferSlice],
    shape: Sequence[Union[PrimExpr, int]],
    dtype: str = "float32",
    data: Var = None,
    strides: Optional[Sequence[int]] = None,
    elem_offset: Optional[int] = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
) -> Buffer: ...
def buffer_decl(
    shape: Sequence[Union[PrimExpr, int]],
    dtype: str = "float32",
    data: Var = None,
    strides: Optional[Sequence[int]] = None,
    elem_offset: Optional[int] = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
) -> Buffer: ...
def alloc_buffer(
    shape: Sequence[Union[PrimExpr, int]],
    dtype: str = "float32",
    data: Var = None,
    strides: Optional[Sequence[int]] = None,
    elem_offset: Optional[int] = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
) -> Buffer: ...

"""
special_stmt - Reads/Writes
"""

def reads(read_regions: Union[BufferSlice, List[BufferSlice]]) -> None: ...
def writes(write_region: Union[BufferSlice, List[BufferSlice]]) -> None: ...
def block_attr(attrs: Mapping[str, Object]) -> None: ...

"""
special_stmt - Axis
"""

class axis:
    @overload
    @staticmethod
    def spatial(dom: Union[PrimExpr, int], value: PrimExpr) -> IterVar: ...
    @overload
    @staticmethod
    def spatial(
        dom: Tuple[Union[PrimExpr, int], Union[PrimExpr, int]], value: PrimExpr
    ) -> IterVar: ...
    @overload
    @staticmethod
    def S(dom: Union[PrimExpr, int], value: PrimExpr) -> IterVar: ...
    @overload
    @staticmethod
    def S(dom: Tuple[Union[PrimExpr, int], Union[PrimExpr, int]], value: PrimExpr) -> IterVar: ...
    @overload
    @staticmethod
    def reduce(dom: Union[PrimExpr, int], value: PrimExpr) -> IterVar: ...
    @overload
    @staticmethod
    def reduce(
        dom: Tuple[Union[PrimExpr, int], Union[PrimExpr, int]], value: PrimExpr
    ) -> IterVar: ...
    @overload
    @staticmethod
    def R(dom: Union[PrimExpr, int], value: PrimExpr) -> IterVar: ...
    @overload
    @staticmethod
    def R(dom: Tuple[Union[PrimExpr, int], Union[PrimExpr, int]], value: PrimExpr) -> IterVar: ...
    @overload
    @staticmethod
    def scan(dom: Union[PrimExpr, int], value: PrimExpr) -> IterVar: ...
    @overload
    @staticmethod
    def scan(
        dom: Tuple[Union[PrimExpr, int], Union[PrimExpr, int]], value: PrimExpr
    ) -> IterVar: ...
    @overload
    @staticmethod
    def opaque(dom: Union[PrimExpr, int], value: PrimExpr) -> IterVar: ...
    @overload
    @staticmethod
    def opaque(
        dom: Tuple[Union[PrimExpr, int], Union[PrimExpr, int]], value: PrimExpr
    ) -> IterVar: ...
    @staticmethod
    def remap(iter_types: str, loop_vars: List[Var]) -> List[IterVar]: ...

def get_axis(begin: PrimExpr, end: PrimExpr, iter_type: int) -> IterVar: ...

"""
special_stmt - Annotations
"""

def buffer_var(dtype: str, storage_scope: str) -> Var: ...
def func_attr(attrs: Mapping[str, Object]) -> None: ...
def prim_func(input_func: Callable) -> PrimFunc: ...

"""
special_stmt - Threads and Bindings
"""

def env_thread(env_name: str) -> IterVar: ...
def bind(iter_var: IterVar, expr: PrimExpr) -> None: ...

"""
Scope handler
"""

class block(ContextManager):
    def __init__(self, name_hint: str = "") -> None: ...
    def __enter__(self) -> Sequence[IterVar]: ...

class init(ContextManager):
    def __init__(self) -> None: ...

class let(ContextManager):
    def __init__(self, var: Var, value: PrimExpr) -> None: ...

def where(cond: PrimExpr) -> None: ...
def allocate(
    extents: List[PrimExpr],
    dtype: str,
    scope: str,
    condition: Union[PrimExpr, builtins.bool] = True,
    annotations: Optional[Mapping[str, Object]] = None,
) -> Var: ...
def launch_thread(env_var: Var, extent: Union[int, PrimExpr]) -> Var: ...
def realize(
    buffer_slice: BufferSlice, scope: str, condition: Union[PrimExpr, builtins.bool] = True
) -> None: ...
def attr(node: PrimExpr, attr_key: str, value: PrimExpr) -> None: ...
def Assert(condition: Union[PrimExpr, builtins.bool], message: str) -> PrimExpr: ...

"""
Scope handler - Loops
"""

def serial(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int],
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def parallel(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int],
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def vectorized(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int],
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def unroll(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int],
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def thread_binding(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int],
    thread: str,
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def for_range(
    begin: Union[PrimExpr, int],
    end: Union[PrimExpr, int] = None,
    annotations: Optional[Mapping[str, Object]] = None,
) -> Iterable[IterVar]: ...
def grid(*extents: Union[PrimExpr, int]) -> Iterable[Sequence[IterVar]]: ...

"""
ty - redefine types
"""

class boolean: ...

class handle:
    @overload
    def __getitem__(self: handle, pos: Sequence[Union[int, PrimExpr, slice]]) -> Buffer: ...
    @overload
    def __getitem__(self: handle, pos: Union[int, PrimExpr, slice]) -> Buffer: ...
    @overload
    def __setitem__(
        self: handle, pos: Sequence[Union[int, PrimExpr, slice]], value: Buffer
    ) -> None: ...
    @overload
    def __setitem__(self: handle, pos: Union[int, PrimExpr, slice], value: Buffer) -> None: ...
    @property
    def data(self: handle) -> Ptr: ...

class Ptr: ...
