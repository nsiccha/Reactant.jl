using Reactant, Test
const IR=Reactant.MLIR.IR
IR.@dispose ctx=Reactant.ReactantContext() begin
    IR.activate(ctx)
    mod=parse(IR.Module,"""
    module {
      func.func @pure(%a: tensor<f64>) -> tensor<f64> {
        %b=stablehlo.multiply %a,%a : tensor<f64>
        return %b : tensor<f64>
      }
      func.func @helper(%a: tensor<f64>) -> tensor<f64> {
        %b=func.call @pure(%a) : (tensor<f64>) -> tensor<f64>
        return %b : tensor<f64>
      }
      func.func @effect(%a: tensor<f64>) -> tensor<f64> {
        %b=stablehlo.custom_call @external(%a) {has_side_effect = true} : (tensor<f64>) -> tensor<f64>
        return %b : tensor<f64>
      }
      func.func @recursive(%a: tensor<f64>) -> tensor<f64> {
        %b=func.call @recursive(%a) : (tensor<f64>) -> tensor<f64>
        return %b : tensor<f64>
      }
    }
    """)
    try
        fns=Dict(String(IR.getattr(op,"sym_name"))=>op for op in first(IR.region(IR.Operation(mod),1)))
        child(name)=first(first(IR.region(fns[name],1)))
        @test Reactant.Compiler._is_replayable_tensor_op(child("pure"),fns)
        @test Reactant.Compiler._is_replayable_tensor_op(child("helper"),fns)
        @test !Reactant.Compiler._is_replayable_tensor_op(child("effect"),fns)
        @test !Reactant.Compiler._is_replayable_tensor_op(child("recursive"),fns)
        before=repr(mod)
        Reactant.Compiler.normalize_condition_loops!(mod)
        @test repr(mod)==before
    finally
        IR.dispose(mod)
        IR.deactivate(ctx)
    end
end
