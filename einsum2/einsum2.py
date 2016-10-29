import autograd
import autograd.numpy as np
from .parallel_matmul import _par_matmul

@autograd.primitive
def batched_dot(a, b, threads=1):
    return _par_matmul(a, b, threads)
batched_dot.defgrad(lambda ans,a,b,threads=1: lambda g: batched_dot(g, np.transpose(b, (0,2,1)),
                                                                    threads))
batched_dot.defgrad(lambda ans,a,b,threads=1: lambda g: batched_dot(np.transpose(a, (0,2,1)), g,
                                                                    threads), argnum=1)

def einsum2(*args, **kwargs):
    """
    einsum2(subscripts_str, arr0, arr1, threads=1, use_dot=True)
    or,
    einsum2(op0, subscript_list0, arr1, subscript_list1,
            output_subscript_list, threads=1, use_dot=True)

    This function is similar to einsum, except it only operates
    on two input arrays, does not allow diagonal operations
    (repeated subscripts on the same array), and requires the output
    subscripts to always be specified.

    Unlike the standard einsum, einsum2 uses a parallel for loop to
    perform computations in parallel. The parameter "threads" controls
    the number of parallel threads to use.

    If use_dot=True, einsum2 will use numpy.dot when possible.
    The "threads" parameter is ignored whenever numpy.dot is used.
    numpy.dot will use BLAS to perform matrix multiplication, which
    can be faster than using the parallel for loop, especially if
    numpy is compiled against a parallel BLAS library such as Intel MKL.
    However, not all operations can be recast into numpy.dot (most notably,
    batched matrix multiplication), and whenever it is not possible
    to use numpy.dot, the parallel for loop will be used.

    The input of einsum2 can be specified in two alternative ways.
    Either the subscripts can be given by a single string, or
    the subscripts of each input array and the output array can each
    be specified in a separate list. See the documentation of numpy.einsum
    for more details.
    """
    if isinstance(args[0], str):
        subscripts, a, b = args[:3]
        ab_subs, out_subs = subscripts.split("->")
        a_subs, b_subs = ab_subs.split(",")
        return _einsum2(a, list(a_subs), b, list(b_subs), list(out_subs), *args[3:], **kwargs)
    else:
        return _einsum2(*args, **kwargs)

def _einsum2(a, a_sublist, b, b_sublist, out_sublist, threads=1, use_dot=True):
    for subs in a_sublist, b_sublist, out_sublist:
        if len(subs) != len(set(subs)):
            raise NotImplementedError("Repeated subscripts not implemented")

    a, a_sublist = _sum_unique_axes(a, a_sublist, b_sublist, out_sublist)
    b, b_sublist = _sum_unique_axes(b, b_sublist, a_sublist, out_sublist)

    a_subs, b_subs, out_subs = map(set, (a_sublist, b_sublist, out_sublist))
    if out_subs - (a_subs | b_subs):
        raise ValueError("Output subscripts must be contained within input subscripts")

    a_minus_b = list(a_subs - b_subs)
    b_minus_a = list(b_subs - a_subs)
    # _sum_unique_axes should have removed any axes unique to a,b
    assert set(a_minus_b) <= out_subs and set(b_minus_a) <= out_subs

    ab = a_subs & b_subs
    abc = list(ab & out_subs)
    ab_minus_c = list(ab - out_subs)

    shapes = {}
    for arr,sublist in ((a,a_sublist), (b,b_sublist)):
        # arr.shape breaks in autograd if it has no dimension
        if sublist:
            for i,s in zip(arr.shape, sublist):
                if s not in shapes:
                    shapes[s] = i
                elif shapes[s] != i:
                    raise ValueError("a,b shapes don't match")

    a = _reshape(a, a_sublist, abc, a_minus_b, ab_minus_c)
    b = _reshape(b, b_sublist, abc, ab_minus_c, b_minus_a)
    if use_dot and not abc:
        c = np.dot(a, b)
    else:
        c = batched_dot(a, b, threads=threads)

    c_sublist = abc + a_minus_b + b_minus_a
    c = np.reshape(c, [shapes[s] for s in c_sublist])

    return _transpose(c, c_sublist, out_sublist)

def einsum1(in_arr, in_sublist, out_sublist):
    in_arr, in_sublist = _sum_unique_axes(in_arr, in_sublist, out_sublist)
    return _transpose(in_arr, in_sublist, out_sublist)

def _reshape(in_arr, in_sublist, *out_sublists):
    assert len(out_sublists) == 3

    old_sublist = in_sublist
    in_sublist = sum(out_sublists, [])
    in_arr = _transpose(in_arr, old_sublist, in_sublist)

    # in_arr.shape breaks in autograd if it has no dimension
    if in_sublist:
        shapes = {s:i for i,s in zip(in_arr.shape, in_sublist)}
    else: shapes = {}
    return np.reshape(in_arr, [np.prod([shapes[s] for s in out_subs], dtype=int)
                               for out_subs in out_sublists])

def _transpose(in_arr, in_sublist, out_sublist):
    if set(in_sublist) != set(out_sublist):
        raise ValueError("Input and output subscripts don't match")
    for sublist in (in_sublist, out_sublist):
        if len(set(sublist)) != len(sublist):
            raise NotImplementedError("Repeated subscripts not implemented")
    in_idxs = {k:v for v,k in enumerate(in_sublist)}
    return np.transpose(in_arr, axes=[in_idxs[s] for s in out_sublist])

def _sum_unique_axes(in_arr, in_sublist, *keep_subs):
    # assume no repeated subscripts
    assert len(in_sublist) == len(set(in_sublist))

    out_sublist = []
    sum_axes = []
    keep_subs = set([s for ks in keep_subs for s in ks])
    for idx, sub in enumerate(in_sublist):
        if sub in keep_subs:
            out_sublist.append(sub)
        else:
            sum_axes.append(idx)
    if sum_axes:
        return np.sum(in_arr, axis=tuple(sum_axes)), out_sublist
    else:
        return in_arr, out_sublist
