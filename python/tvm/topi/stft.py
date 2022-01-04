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
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks
"""STFT operator"""
import tvm
from tvm import te
from ..tir import decl_buffer, ir_builder, Cast, sin, cos
from ..te import extern, div, floordiv, floormod
from .utils import ceil_div
import math

def stft(
    data,
    n_fft,
    hop_length,
    win_length,
    window,
    output_shape,
):

    def gen_ir(
        data_ptr,
        n_fft_ptr,
        hop_length_ptr,
        win_length_ptr, 
        window_ptr,
        output_ptr,
    ):
        ib = ir_builder.create()

        data = ib.buffer_ptr(data_ptr)
        n_fft = ib.buffer_ptr(n_fft_ptr)

        hop_length = ib.buffer_ptr(hop_length_ptr)
        win_length = ib.buffer_ptr(win_length_ptr)
        window = ib.buffer_ptr(window_ptr)
        output = ib.buffer_ptr(output_ptr)

        with ib.for_range(0, output_ptr.shape[0]) as batch:
            with ib.for_range(0, output_ptr.shape[1]) as row:
                with ib.for_range(0, output_ptr.shape[2]) as col:
                    # with ib.for_range(0, output_ptr.shape[3]) as l:
                    #     output[i, j, k, l] = Cast(data_ptr.dtype, n_fft[0])
                    output[batch, row, col, 0] = Cast(data_ptr.dtype, 0)
                    output[batch, row, col, 1] = Cast(data_ptr.dtype, 0) 
                    with ib.for_range(0, win_length[0]) as wlen:
                        output[batch, row, col, 0] += (window[wlen] * data[batch, col*hop_length[0]+wlen] * cos(2*math.pi*row*wlen/win_length[0]))
                        output[batch, row, col, 1] -= (window[wlen] * data[batch, col*hop_length[0]+wlen] * sin(2*math.pi*row*wlen/win_length[0]))

        return ib.get()

    output_buf = decl_buffer(output_shape, data.dtype, "output_buf")

    return extern(
        [output_shape],
        [data, n_fft, hop_length, win_length, window],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], ins[3], ins[4], outs[0]),
        dtype=[data.dtype],
        out_buffers=[output_buf],
        name="stft_cpu",
        tag="stft_cpu",
    )
