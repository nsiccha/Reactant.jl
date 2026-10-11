# Runtime condition loops have no static tape size. For pure tensor programs,
# first count their actual iterations, then replay the unchanged body in a
# canonical counted loop. Ordinary Enzyme reverse can apply bounded binomial
# checkpointing to that replay. The original condition and lazy branches are
# preserved; no iteration is expanded according to data or shapes.

function _condition_loop_postwalk!(f, op)
    for region in op, block in region, child in collect(block)
        _condition_loop_postwalk!(f, child)
    end
    f(op)
end

# Recognize the canonical unit-step loop emitted by @trace for. A different
# pure while may still be counted safely by the general transformation below.
function _is_unit_counted_loop(op)
    cond = first(MLIR.IR.region(op, 1))
    body = first(MLIR.IR.region(op, 2))
    MLIR.IR.nargs(body) > 0 || return false
    iv = MLIR.IR.argument(body, 1)
    MLIR.IR.type(iv) == MLIR.IR.TensorType(Int[], MLIR.IR.Type(Int64)) || return false
    next = MLIR.IR.operand(MLIR.IR.terminator(body), 1)
    MLIR.IR.is_op_res(next) || return false
    add = MLIR.IR.op_owner(next)
    MLIR.IR.name(add) == "stablehlo.add" || return false
    MLIR.IR.noperands(add) == 2 || return false
    lhs, rhs = MLIR.IR.operand(add, 1), MLIR.IR.operand(add, 2)
    step = lhs == iv ? rhs : rhs == iv ? lhs : nothing
    step === nothing && return false
    MLIR.IR.is_op_res(step) || return false
    constant = MLIR.IR.op_owner(step)
    MLIR.IR.name(constant) == "stablehlo.constant" || return false
    MLIR.IR.getattr(constant, "value") == MLIR.IR.DenseElementsAttribute(fill(Int64(1))) ||
        return false
    pred = MLIR.IR.operand(MLIR.IR.terminator(cond), 1)
    MLIR.IR.is_op_res(pred) || return false
    cmp = MLIR.IR.op_owner(pred)
    MLIR.IR.name(cmp) == "stablehlo.compare" || return false
    MLIR.IR.getattr(cmp, "comparison_direction") ==
        parse(MLIR.IR.Attribute, "#stablehlo<comparison_direction LT>") || return false
    MLIR.IR.operand(cmp, 1) == MLIR.IR.argument(cond, 1) || return false
    limit = MLIR.IR.operand(cmp, 2)
    if MLIR.IR.is_block_arg(limit) && MLIR.IR.owner(limit) == cond
        i = MLIR.IR.block_arg_num(limit) + 1
        i > 1 || return false
        return MLIR.IR.operand(MLIR.IR.terminator(body), i) == MLIR.IR.argument(body, i)
    end
    return !MLIR.IR.is_op_res(limit) || MLIR.IR.block(MLIR.IR.op_owner(limit)) != cond
end

function _copy_loop_regions(op)
    clone = copy(op)
    regions = [MLIR.IR.Region(), MLIR.IR.Region()]
    for i in 1:2
        MLIR.API.mlirRegionTakeBody(regions[i], MLIR.IR.region(clone, i))
    end
    MLIR.IR.dispose(clone)
    return regions
end

function _replace_loop_return!(block, values)
    old = MLIR.IR.terminator(block)
    new = MLIR.IR.create_operation_common("stablehlo.return", MLIR.IR.location(old);
        results=MLIR.IR.Type[], operands=values)
    MLIR.IR.insert_before!(block, old, new)
    MLIR.IR.rmfromparent!(old)
    MLIR.IR.dispose(old)
end

function _loop_constant_before(op, n)
    c = MLIR.IR.create_operation_common("stablehlo.constant", MLIR.IR.location(op);
        results=[MLIR.IR.TensorType(Int[], MLIR.IR.Type(Int64))],
        attributes=[MLIR.IR.NamedAttribute("value", MLIR.IR.DenseElementsAttribute(fill(Int64(n))))])
    MLIR.IR.insert_before!(MLIR.IR.block(op), op, c)
    return MLIR.IR.result(c, 1)
end

function _normalize_condition_loop!(op)
    MLIR.IR.name(op) == "stablehlo.while" || return
    MLIR.IR.getattr(op, "enzyme.binomial_checkpointing") === nothing && return
    MLIR.IR.getattr(op, "reactant.iteration_count") !== nothing && return
    _is_unit_counted_loop(op) && return
    loc = MLIR.IR.location(op)
    zero = _loop_constant_before(op, 0)
    one = _loop_constant_before(op, 1)
    ty = MLIR.IR.type(zero)
    args = [MLIR.IR.operand(op, i) for i in 1:MLIR.IR.noperands(op)]
    types = [MLIR.IR.type(MLIR.IR.result(op, i)) for i in 1:MLIR.IR.nresults(op)]
    attrs = [MLIR.IR.getattr(op, i) for i in 1:MLIR.IR.nattrs(op)]
    count_regions = _copy_loop_regions(op)
    for region in count_regions
        MLIR.IR.push_argument!(first(region), ty; location=loc)
    end
    block = first(count_regions[2])
    term = MLIR.IR.terminator(block)
    values = [MLIR.IR.operand(term, i) for i in 1:MLIR.IR.noperands(term)]
    add = MLIR.IR.create_operation_common("stablehlo.add", loc;
        results=[ty], operands=[MLIR.IR.argument(block, MLIR.IR.nargs(block)), one])
    MLIR.IR.insert_before!(block, term, add)
    _replace_loop_return!(block, [values; MLIR.IR.result(add, 1)])
    counter = MLIR.IR.create_operation_common("stablehlo.while", loc;
        results=[types; ty], operands=[args; zero], owned_regions=count_regions,
        attributes=attrs)
    MLIR.IR.setattr!(counter, "reactant.iteration_count", MLIR.IR.UnitAttribute())
    MLIR.IR.insert_before!(MLIR.IR.block(op), op, counter)
    count = MLIR.IR.result(counter, MLIR.IR.nresults(counter))

    replay_regions = _copy_loop_regions(op)
    for region in replay_regions
        MLIR.API.mlirBlockInsertArgument(first(region), 0, ty, loc)
        MLIR.API.mlirBlockInsertArgument(first(region), 1, ty, loc)
    end
    cond = first(replay_regions[1])
    for child in reverse(collect(cond))
        MLIR.IR.rmfromparent!(child); MLIR.IR.dispose(child)
    end
    compare = MLIR.IR.create_operation_common("stablehlo.compare", loc;
        results=[MLIR.IR.TensorType(Int[], MLIR.IR.Type(Bool))],
        operands=[MLIR.IR.argument(cond, 1), MLIR.IR.argument(cond, 2)],
        attributes=[MLIR.IR.NamedAttribute("comparison_direction",
            parse(MLIR.IR.Attribute, "#stablehlo<comparison_direction LT>"))])
    push!(cond, compare)
    push!(cond, MLIR.IR.create_operation_common("stablehlo.return", loc;
        results=MLIR.IR.Type[], operands=[MLIR.IR.result(compare, 1)]))
    block = first(replay_regions[2])
    term = MLIR.IR.terminator(block)
    values = [MLIR.IR.operand(term, i) for i in 1:MLIR.IR.noperands(term)]
    add = MLIR.IR.create_operation_common("stablehlo.add", loc;
        results=[ty], operands=[MLIR.IR.argument(block, 1), one])
    MLIR.IR.insert_before!(block, term, add)
    _replace_loop_return!(block, [MLIR.IR.result(add, 1); MLIR.IR.argument(block, 2); values])
    replay = MLIR.IR.create_operation_common("stablehlo.while", loc;
        results=[ty; ty; types], operands=[zero; count; args], owned_regions=replay_regions,
        attributes=attrs)
    MLIR.IR.insert_before!(MLIR.IR.block(op), op, replay)
    for i in 1:MLIR.IR.nresults(op)
        MLIR.API.mlirValueReplaceAllUsesOfWith(MLIR.IR.result(op, i), MLIR.IR.result(replay, i+2))
    end
    MLIR.IR.rmfromparent!(op); MLIR.IR.dispose(op)
end


function _condition_loop_callee(op)
    attr = MLIR.IR.getattr(op, MLIR.IR.name(op) == "func.call" ? "callee" : "fn")
    attr === nothing && return nothing
    MLIR.IR.issymbolref(attr) || return nothing
    MLIR.IR.nnestedrefs(attr) == 0 || return nothing
    return MLIR.IR.rootref(attr)
end

# Fail closed for unknown dialects, opaque calls, communication and stateful
# RNG. These operations cannot in general be replayed without observable effects.
function _is_replayable_tensor_op(op, functions, visiting=Set{String}())
    name = MLIR.IR.name(op)
    if name == "func.call" || name == "enzyme.batch"
        callee = _condition_loop_callee(op)
        callee === nothing && return false
        callee in visiting && return false
        haskey(functions, callee) || return false
        fn = functions[callee]
        MLIR.IR.nregions(fn) > 0 || return false
        isempty(first(MLIR.IR.region(fn, 1))) && return false
        push!(visiting, callee)
        result = all(child -> _is_replayable_tensor_op(child, functions, visiting),
            (child for region in fn for block in region for child in block))
        delete!(visiting, callee)
        return result
    end
    allowed = startswith(name, "stablehlo.") && name ∉ (
        "stablehlo.custom_call", "stablehlo.infeed", "stablehlo.outfeed",
        "stablehlo.send", "stablehlo.recv", "stablehlo.rng",
        "stablehlo.create_token", "stablehlo.after_all", "stablehlo.all_gather",
        "stablehlo.all_reduce", "stablehlo.all_to_all", "stablehlo.reduce_scatter",
        "stablehlo.collective_permute", "stablehlo.collective_broadcast",
    )
    (allowed || name == "func.return") || return false
    return all(child -> _is_replayable_tensor_op(child, functions, visiting),
        (child for region in op for block in region for child in block))
end

function normalize_condition_loops!(mod)
    root = MLIR.IR.Operation(mod)
    functions = Dict{String,MLIR.IR.Operation}()
    targets = Set{String}()
    regions = MLIR.IR.Operation[]
    _condition_loop_postwalk!(root) do op
        name = MLIR.IR.name(op)
        if name == "func.func"
            functions[String(MLIR.IR.getattr(op, "sym_name"))] = op
        elseif name == "enzyme.autodiff"
            callee = _condition_loop_callee(op)
            callee === nothing || push!(targets, callee)
        elseif name == "enzyme.autodiff_region"
            push!(regions, op)
        end
    end
    # Include helper functions differentiated through ordinary calls/batches.
    visited = Set{String}()
    while !isempty(targets)
        name = pop!(targets)
        name in visited && continue
        push!(visited, name)
        haskey(functions, name) || continue
        fn = functions[name]
        _condition_loop_postwalk!(fn) do op
            if MLIR.IR.name(op) in ("func.call", "enzyme.batch")
                callee = _condition_loop_callee(op)
                callee === nothing || push!(targets, callee)
            end
        end
        push!(regions, fn)
    end
    for region in regions
        _condition_loop_postwalk!(region) do op
            MLIR.IR.name(op) == "stablehlo.while" || return
            _is_replayable_tensor_op(op, functions) || return
            _normalize_condition_loop!(op)
        end
    end
    return mod
end
