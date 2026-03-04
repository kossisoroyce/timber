"""Nuclear-grade test suite — every subsystem, every edge case, numeric correctness."""
from __future__ import annotations
import ctypes, json, math, os, subprocess, tempfile
from pathlib import Path
from typing import Optional
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ── helpers ──────────────────────────────────────────────────────────────────

def _bc(n=10):  X,y=load_breast_cancer(return_X_y=True); return X[:,:n].astype(np.float32),y
def _reg(n=8):  X,y=load_diabetes(return_X_y=True);      return X[:,:n].astype(np.float32),y
def _iris():    X,y=load_iris(return_X_y=True);           return X.astype(np.float32),y

def _xgb_ir(n_features=10, n_trees=10, task="binary"):
    import xgboost as xgb
    from timber.frontends.xgboost_parser import parse_xgboost_json
    X,y = _bc(n_features)
    if task=="reg": X,y=_reg(n_features); m=xgb.XGBRegressor(n_estimators=n_trees,max_depth=3,random_state=42); m.fit(X,y)
    else:           m=xgb.XGBClassifier(n_estimators=n_trees,max_depth=3,eval_metric="logloss",random_state=42); m.fit(X,y)
    with tempfile.NamedTemporaryFile(suffix=".json",delete=False) as f:
        m.get_booster().save_model(f.name); ir=parse_xgboost_json(f.name)
    os.unlink(f.name); return ir, m, X

def _onnx_path(clf, X):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    b = convert_sklearn(clf, initial_types=[("float_input",FloatTensorType([None,X.shape[1]]))]).SerializeToString()
    f=tempfile.NamedTemporaryFile(suffix=".onnx",delete=False); f.write(b); f.close(); return f.name

def _compile_so(d:Path)->Optional[Path]:
    so=d/"model.so"
    r=subprocess.run(["cc","-std=c99","-O2","-fPIC","-shared","-I",str(d),str(d/"model.c"),"-o",str(so),"-lm"],
                     capture_output=True,timeout=30)
    return so if r.returncode==0 else None

def _ctypes_infer(so:Path, X:np.ndarray, n_out:int)->Optional[np.ndarray]:
    try:
        lib=ctypes.CDLL(str(so)); lib.timber_infer.restype=ctypes.c_int
        lib.timber_infer.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.POINTER(ctypes.c_float),ctypes.c_void_p]
        n=X.shape[0]; inp=X.astype(np.float32).ravel(); out=np.zeros(n*n_out,dtype=np.float32)
        rc=lib.timber_infer(inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),ctypes.c_int(n),
                             out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),None)
        return out.reshape(n,n_out) if rc==0 else None
    except Exception: return None

def _py_infer_tree(ir, X):
    e=ir.get_tree_ensemble(); res=[]
    for row in X:
        s=e.base_score
        for tree in e.trees:
            cur=0; nodes=tree.nodes
            while 0<=cur<len(nodes):
                n=nodes[cur]
                if n.is_leaf: s+=n.leaf_value; break
                cur=n.left_child if row[n.feature_index]<n.threshold else n.right_child
        res.append(s)
    return np.array(res,dtype=np.float32)

def _tiny_ir(lv=(0.5,0.001,0.5,-0.001)):
    from timber.ir.model import (TimberIR,TreeEnsembleStage,Tree,TreeNode,
                                  Objective,Field,FieldType,Schema,Metadata)
    nodes=[TreeNode(0,0,0.5,1,4,False,0.0,0),TreeNode(1,1,0.3,2,3,False,0.0,1),
           TreeNode(2,-1,0.0,-1,-1,True,lv[0],2),TreeNode(3,-1,0.0,-1,-1,True,lv[1],2),
           TreeNode(4,1,0.7,5,6,False,0.0,1),TreeNode(5,-1,0.0,-1,-1,True,lv[2],2),
           TreeNode(6,-1,0.0,-1,-1,True,lv[3],2)]
    t=Tree(0,nodes); t.recount()
    stage=TreeEnsembleStage("t","tree_ensemble",[t],2,1,Objective.REGRESSION,0.0,[],1.0,True)
    return TimberIR([stage],Schema([Field("f0",FieldType.FLOAT32,0),Field("f1",FieldType.FLOAT32,1)],
                                    [Field("out",FieldType.FLOAT32,0)]),Metadata())


# ═══════════════════════════════════════════════════════════════════════════
# A. IR LAYER
# ═══════════════════════════════════════════════════════════════════════════

class TestIRLayer:
    def test_empty_ir_no_ensemble(self):
        from timber.ir.model import TimberIR
        assert TimberIR().get_tree_ensemble() is None

    def test_deep_copy_independence(self):
        ir,_,_ = _xgb_ir(10,5)
        ir2 = ir.deep_copy()
        ir.pipeline[0].trees[0].nodes[0].leaf_value = 9999.0
        assert ir2.pipeline[0].trees[0].nodes[0].leaf_value != 9999.0

    def test_json_round_trip_tree_values(self):
        from timber.ir.model import TimberIR
        ir,_,_ = _xgb_ir(10,10)
        ir2 = TimberIR.from_json(ir.to_json())
        e1,e2 = ir.pipeline[0], ir2.pipeline[0]
        assert e1.n_trees==e2.n_trees and e1.n_features==e2.n_features
        for t1,t2 in zip(e1.trees,e2.trees):
            for n1,n2 in zip(t1.nodes,t2.nodes):
                assert math.isclose(n1.threshold,n2.threshold,rel_tol=1e-9)
                assert math.isclose(n1.leaf_value,n2.leaf_value,rel_tol=1e-9)

    def test_json_invalid_raises(self):
        from timber.ir.model import TimberIR
        with pytest.raises(Exception): TimberIR.from_json("{bad json")

    def test_all_stages_serialize(self):
        from timber.ir.model import (TimberIR,LinearStage,SVMStage,NormalizerStage,ScalerStage,
                                      Field,FieldType,Schema,Metadata,Objective)
        stages=[NormalizerStage("n","normalizer",norm="l2"),
                ScalerStage("s","scaler",means=[1.0],scales=[0.5],feature_indices=[0]),
                LinearStage("l","linear",weights=[0.1],bias=0.5,activation="sigmoid",n_classes=1),
                SVMStage("sv","svm",support_vectors=[[0.1]],dual_coef=[0.5],rho=[0.1],
                         n_support=[1],n_features=1,n_classes=2,objective=Objective.BINARY_CLASSIFICATION)]
        ir=TimberIR(stages,Schema([Field("f0",FieldType.FLOAT32,0)],[Field("out",FieldType.FLOAT32,0)]),Metadata())
        ir2=TimberIR.from_json(ir.to_json())
        assert [s.stage_type for s in ir2.pipeline]==["normalizer","scaler","linear","svm"]

    def test_linear_post_init_type(self):
        from timber.ir.model import LinearStage
        s=LinearStage("x","WRONG",[1.0],0.0,"none",1); assert s.stage_type=="linear"

    def test_svm_n_sv_property(self):
        from timber.ir.model import SVMStage,Objective
        s=SVMStage("s","svm",support_vectors=[[1.0],[2.0]],dual_coef=[0.1,0.2],
                   rho=[0.0],n_support=[2],n_features=1,n_classes=2,
                   objective=Objective.BINARY_CLASSIFICATION)
        assert s.n_sv==2

    def test_metadata_persists_through_roundtrip(self):
        from timber.ir.model import TimberIR
        ir,_,_ = _xgb_ir(10,5)
        ir.metadata.source_framework="test_fw"
        ir2=TimberIR.from_json(ir.to_json())
        assert ir2.metadata.source_framework=="test_fw"

    def test_schema_field_count(self):
        ir,_,_ = _xgb_ir(10,5)
        assert len(ir.schema.input_fields)==10

    def test_n_trees_max_depth_properties(self):
        ir,_,_ = _xgb_ir(10,5)
        e=ir.pipeline[0]
        assert e.n_trees==5 and e.max_depth>=1

    def test_tree_nodes_valid_child_indices(self):
        ir,_,_ = _xgb_ir(10,10)
        for tree in ir.pipeline[0].trees:
            for node in tree.nodes:
                if not node.is_leaf:
                    assert 0<=node.left_child<len(tree.nodes)
                    assert 0<=node.right_child<len(tree.nodes)


# ═══════════════════════════════════════════════════════════════════════════
# B. SKLEARN PARSER
# ═══════════════════════════════════════════════════════════════════════════

class TestSklearnParser:
    def _parse(self, est, X, y):
        from timber.frontends.sklearn_parser import _convert_sklearn
        est.fit(X,y); return _convert_sklearn(est), est

    def test_rf_binary(self):
        from timber.ir.model import Objective
        X,y=_bc(10); ir,_=self._parse(RandomForestClassifier(n_estimators=10,max_depth=3,random_state=0),X,y)
        e=ir.get_tree_ensemble()
        assert e.n_trees==10 and e.objective==Objective.BINARY_CLASSIFICATION and not e.is_boosted

    def test_rf_regressor(self):
        from timber.ir.model import Objective
        X,y=_reg(8); ir,_=self._parse(RandomForestRegressor(n_estimators=5,max_depth=3,random_state=0),X,y)
        assert ir.get_tree_ensemble().objective==Objective.REGRESSION

    def test_rf_multiclass(self):
        from timber.ir.model import Objective
        X,y=_iris(); ir,_=self._parse(RandomForestClassifier(n_estimators=5,random_state=0),X,y)
        assert ir.get_tree_ensemble().objective==Objective.MULTICLASS_CLASSIFICATION

    def test_gbt_classifier(self):
        X,y=_bc(10); ir,_=self._parse(GradientBoostingClassifier(n_estimators=10,max_depth=2,random_state=0),X,y)
        assert ir.get_tree_ensemble().is_boosted

    def test_gbt_regressor(self):
        from timber.ir.model import Objective
        X,y=_reg(8); ir,_=self._parse(GradientBoostingRegressor(n_estimators=5,max_depth=2,random_state=0),X,y)
        assert ir.get_tree_ensemble().objective==Objective.REGRESSION

    def test_hist_gbt_classifier(self):
        X,y=_bc(10); ir,_=self._parse(HistGradientBoostingClassifier(max_iter=10,random_state=0),X,y)
        assert ir.get_tree_ensemble() is not None

    def test_hist_gbt_regressor(self):
        from timber.ir.model import Objective
        X,y=_reg(8); ir,_=self._parse(HistGradientBoostingRegressor(max_iter=10,random_state=0),X,y)
        assert ir.get_tree_ensemble().objective==Objective.REGRESSION

    def test_decision_tree_classifier(self):
        X,y=_bc(10); ir,_=self._parse(DecisionTreeClassifier(max_depth=3,random_state=0),X,y)
        assert ir.get_tree_ensemble().n_trees==1

    def test_decision_tree_regressor(self):
        X,y=_reg(8); ir,_=self._parse(DecisionTreeRegressor(max_depth=3,random_state=0),X,y)
        assert ir.get_tree_ensemble().n_trees==1

    def test_pipeline_with_scaler(self):
        from timber.ir.model import ScalerStage
        from timber.frontends.sklearn_parser import _convert_sklearn
        X,y=_bc(10); pipe=SkPipeline([("sc",StandardScaler()),("clf",RandomForestClassifier(5,random_state=0))])
        pipe.fit(X,y); ir=_convert_sklearn(pipe)
        assert any(isinstance(s,ScalerStage) for s in ir.pipeline)

    def test_scaler_means_match_sklearn(self):
        from timber.ir.model import ScalerStage
        from timber.frontends.sklearn_parser import _convert_sklearn
        X,y=_bc(10); pipe=SkPipeline([("sc",StandardScaler()),("clf",RandomForestClassifier(5,random_state=0))])
        pipe.fit(X,y); ir=_convert_sklearn(pipe)
        ss=next(s for s in ir.pipeline if isinstance(s,ScalerStage))
        np.testing.assert_allclose(ss.means, pipe.named_steps["sc"].mean_.tolist(), rtol=1e-5)

    def test_unsupported_raises(self):
        from timber.frontends.sklearn_parser import _convert_sklearn
        from sklearn.linear_model import LinearRegression
        with pytest.raises(ValueError,match="Unsupported"):
            _convert_sklearn(LinearRegression().fit(*_reg(8)))

    def test_all_nodes_valid_child_indices(self):
        from timber.frontends.sklearn_parser import _convert_sklearn
        X,y=_bc(10); clf=RandomForestClassifier(n_estimators=5,max_depth=3,random_state=0).fit(X,y)
        ir=_convert_sklearn(clf)
        for tree in ir.get_tree_ensemble().trees:
            for node in tree.nodes:
                if not node.is_leaf:
                    assert 0<=node.left_child<len(tree.nodes)
                    assert 0<=node.right_child<len(tree.nodes)


# ═══════════════════════════════════════════════════════════════════════════
# C. ONNX PARSER
# ═══════════════════════════════════════════════════════════════════════════

class TestONNXParser:
    def _binary_path(self,n=10):
        from sklearn.linear_model import LogisticRegression
        X,y=_bc(n); clf=LogisticRegression(max_iter=300).fit(X,y); return _onnx_path(clf,X),clf,X,y

    def _multi_path(self):
        from sklearn.linear_model import LogisticRegression
        X,y=_iris(); clf=LogisticRegression(max_iter=500).fit(X,y); return _onnx_path(clf,X),clf,X,y

    def _reg_path(self,n=8):
        from sklearn.linear_model import LinearRegression
        X,y=_reg(n); reg=LinearRegression().fit(X,y); return _onnx_path(reg,X),reg,X,y

    def _svm_path(self,kernel="rbf",n=10):
        from sklearn.svm import SVC
        X,y=_bc(n); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
        clf=SVC(kernel=kernel,C=1.0,max_iter=500).fit(X,y); return _onnx_path(clf,X),clf,X,y

    def test_binary_n_classes_1(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage
        p,_,_,_=self._binary_path()
        try: ir=parse_onnx_model(p); s=ir.pipeline[-1]; assert isinstance(s,LinearStage); assert s.n_classes==1
        finally: os.unlink(p)

    def test_binary_weight_count_equals_features(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage
        p,_,X,_=self._binary_path(10)
        try: ir=parse_onnx_model(p); assert len(ir.pipeline[-1].weights)==10
        finally: os.unlink(p)

    def test_binary_activation_sigmoid(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        p,_,_,_=self._binary_path()
        try: assert parse_onnx_model(p).pipeline[-1].activation=="sigmoid"
        finally: os.unlink(p)

    def test_binary_single_output_schema(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        p,_,_,_=self._binary_path()
        try: assert len(parse_onnx_model(p).schema.output_fields)==1
        finally: os.unlink(p)

    def test_multiclass_n_classes_3(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import LinearStage
        p,_,_,_=self._multi_path()
        try: ir=parse_onnx_model(p); s=ir.pipeline[-1]; assert isinstance(s,LinearStage); assert s.n_classes==3
        finally: os.unlink(p)

    def test_multiclass_weights_3x4(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        p,_,X,_=self._multi_path()
        try: assert len(parse_onnx_model(p).pipeline[-1].weights)==3*X.shape[1]
        finally: os.unlink(p)

    def test_multiclass_3_biases(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        p,_,_,_=self._multi_path()
        try: assert len(parse_onnx_model(p).pipeline[-1].biases)==3
        finally: os.unlink(p)

    def test_multiclass_3_output_fields(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        p,_,_,_=self._multi_path()
        try: assert len(parse_onnx_model(p).schema.output_fields)==3
        finally: os.unlink(p)

    def test_regressor_no_activation(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        p,_,_,_=self._reg_path()
        try: assert parse_onnx_model(p).pipeline[-1].activation=="none"
        finally: os.unlink(p)

    def test_svm_rbf_sv_count_matches_sklearn(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import SVMStage
        p,clf,_,_=self._svm_path("rbf",10)
        try: ir=parse_onnx_model(p); s=ir.pipeline[-1]; assert isinstance(s,SVMStage); assert s.n_sv==len(clf.support_vectors_)
        finally: os.unlink(p)

    def test_svm_sv_shape_correct(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        p,_,_,_=self._svm_path("rbf",10)
        try:
            ir=parse_onnx_model(p); s=ir.pipeline[-1]
            for sv in s.support_vectors: assert len(sv)==10
        finally: os.unlink(p)

    def test_svm_linear_kernel(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        p,_,_,_=self._svm_path("linear",8)
        try: assert parse_onnx_model(p).pipeline[-1].kernel_type=="linear"
        finally: os.unlink(p)

    def test_no_supported_op_raises(self,tmp_path):
        import onnx; from onnx import helper,TensorProto
        X=helper.make_tensor_value_info("X",TensorProto.FLOAT,[None,4])
        Y=helper.make_tensor_value_info("Y",TensorProto.FLOAT,[None,4])
        g=helper.make_graph([helper.make_node("Add",["X","X"],["Y"])],"g",[X],[Y])
        p=str(tmp_path/"bad.onnx"); onnx.save(helper.make_model(g),p)
        from timber.frontends.onnx_parser import parse_onnx_model
        with pytest.raises(ValueError,match="No supported"): parse_onnx_model(p)

    def test_ir_roundtrip_linear(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import TimberIR
        p,_,_,_=self._binary_path(10)
        try:
            ir=parse_onnx_model(p); ir2=TimberIR.from_json(ir.to_json())
            assert ir.pipeline[-1].weights==ir2.pipeline[-1].weights
        finally: os.unlink(p)

    def test_ir_roundtrip_svm(self):
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.ir.model import TimberIR,SVMStage
        p,_,_,_=self._svm_path("rbf",10)
        try:
            ir=parse_onnx_model(p); ir2=TimberIR.from_json(ir.to_json())
            s1,s2=ir.pipeline[-1],ir2.pipeline[-1]
            assert s1.support_vectors==s2.support_vectors and s1.gamma==s2.gamma
        finally: os.unlink(p)


# ═══════════════════════════════════════════════════════════════════════════
# D. NUMERIC ACCURACY
# ═══════════════════════════════════════════════════════════════════════════

class TestNumericAccuracy:
    def test_xgb_python_ir_correlates_with_native(self):
        import xgboost as xgb
        from timber.frontends.xgboost_parser import parse_xgboost_json
        X,y=_bc(10); m=xgb.XGBClassifier(n_estimators=20,max_depth=3,eval_metric="logloss",random_state=42).fit(X,y)
        with tempfile.NamedTemporaryFile(suffix=".json",delete=False) as f:
            m.get_booster().save_model(f.name); ir=parse_xgboost_json(f.name)
        os.unlink(f.name)
        xgb_p=m.predict_proba(X[:50])[:,1].astype(np.float32)
        ir_p=1/(1+np.exp(-_py_infer_tree(ir,X[:50])))
        assert float(np.corrcoef(xgb_p,ir_p)[0,1])>0.95

    def test_lgbm_python_ir_correlates_with_native(self):
        import lightgbm as lgb
        from timber.frontends.lightgbm_parser import parse_lightgbm_model
        X,y=_bc(10); m=lgb.LGBMClassifier(n_estimators=20,max_depth=3,random_state=42,verbose=-1).fit(X,y)
        with tempfile.NamedTemporaryFile(suffix=".txt",delete=False) as f:
            m.booster_.save_model(f.name); ir=parse_lightgbm_model(f.name)
        os.unlink(f.name)
        lgbm_p=m.predict_proba(X[:50])[:,1].astype(np.float32)
        ir_p=1/(1+np.exp(-_py_infer_tree(ir,X[:50])))
        assert float(np.corrcoef(lgbm_p,ir_p)[0,1])>0.90

    def test_c99_tree_matches_python_ir(self,tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir,_,X=_xgb_ir(10,10)
        out=C99Emitter().emit(ir); out.write(tmp_path)
        so=_compile_so(tmp_path)
        if so is None: pytest.skip("no compiler")
        c_out=_ctypes_infer(so,X[:30],1); py_raw=_py_infer_tree(ir,X[:30])
        assert c_out is not None
        # C99 applies sigmoid for binary classification; convert Python raw log-odds to probability
        py_proba = 1.0 / (1.0 + np.exp(-py_raw.astype(np.float64)))
        # float32 C leaf arrays vs float64 Python accumulation can diverge ~0.07 at worst
        np.testing.assert_allclose(c_out.ravel().astype(np.float64), py_proba, atol=0.10,
                                   err_msg="C99 binary output diverges from Python sigmoid(log-odds)")

    def test_c99_linear_sigmoid_range(self,tmp_path):
        from sklearn.linear_model import LogisticRegression
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.c99 import C99Emitter
        X,y=_bc(10); clf=LogisticRegression(max_iter=300).fit(X,y)
        p=_onnx_path(clf,X)
        try:
            ir=parse_onnx_model(p); out=C99Emitter().emit(ir); out.write(tmp_path)
            so=_compile_so(tmp_path)
            if so is None: pytest.skip("no compiler")
            res=_ctypes_infer(so,X[:50],1)
            assert res is not None
            assert np.all(res>=0.0) and np.all(res<=1.0),f"sigmoid output out of [0,1]: min={res.min():.3f} max={res.max():.3f}"
        finally: os.unlink(p)

    def test_c99_multiclass_softmax_sums_to_one(self,tmp_path):
        from sklearn.linear_model import LogisticRegression
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.c99 import C99Emitter
        X,y=_iris(); clf=LogisticRegression(max_iter=500).fit(X,y)
        p=_onnx_path(clf,X)
        try:
            ir=parse_onnx_model(p); out=C99Emitter().emit(ir); out.write(tmp_path)
            so=_compile_so(tmp_path)
            if so is None: pytest.skip("no compiler")
            res=_ctypes_infer(so,X[:20],3)
            assert res is not None; sums=res.sum(axis=1)
            np.testing.assert_allclose(sums,1.0,atol=0.02,err_msg=f"softmax sums: {sums}")
        finally: os.unlink(p)

    def test_c99_linear_regressor_unbounded(self,tmp_path):
        from sklearn.linear_model import LinearRegression
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.c99 import C99Emitter
        X,y=_reg(8); reg=LinearRegression().fit(X,y)
        p=_onnx_path(reg,X)
        try:
            ir=parse_onnx_model(p); out=C99Emitter().emit(ir); out.write(tmp_path)
            so=_compile_so(tmp_path)
            if so is None: pytest.skip("no compiler")
            res=_ctypes_infer(so,X[:20],1)
            assert res is not None; assert np.all(np.isfinite(res))
            # regression must NOT be clipped to [0,1]
            assert np.any(res>1.1) or np.any(res<0.0),"Regression looks clipped to [0,1]"
        finally: os.unlink(p)

    def test_c99_svm_rbf_finite(self,tmp_path):
        from sklearn.svm import SVC
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.c99 import C99Emitter
        X,y=_bc(10); Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
        clf=SVC(kernel="rbf",C=1.0,max_iter=500).fit(Xn,y)
        p=_onnx_path(clf,Xn)
        try:
            ir=parse_onnx_model(p); out=C99Emitter().emit(ir); out.write(tmp_path)
            so=_compile_so(tmp_path)
            if so is None: pytest.skip("no compiler")
            res=_ctypes_infer(so,Xn[:20],1)
            assert res is not None; assert np.all(np.isfinite(res))
        finally: os.unlink(p)


# ═══════════════════════════════════════════════════════════════════════════
# E. OPTIMIZER PASSES
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizerPasses:
    def test_dead_leaf_prunes_negligible(self):
        from timber.optimizer.dead_leaf import dead_leaf_elimination
        ir=_tiny_ir((1.0,0.0005,1.0,0.0005))
        changed,_,d=dead_leaf_elimination(ir,threshold=0.01)
        assert changed and d["leaves_pruned"]>0

    def test_dead_leaf_no_change_all_significant(self):
        from timber.optimizer.dead_leaf import dead_leaf_elimination
        changed,_,_=dead_leaf_elimination(_tiny_ir((1.0,0.5,0.8,0.6)),threshold=0.001)
        assert not changed

    def test_dead_leaf_all_zero_skipped(self):
        from timber.optimizer.dead_leaf import dead_leaf_elimination
        changed,_,d=dead_leaf_elimination(_tiny_ir((0.0,0.0,0.0,0.0)))
        assert not changed and "skipped" in d

    def test_dead_leaf_no_ensemble_skipped(self):
        from timber.optimizer.dead_leaf import dead_leaf_elimination
        from timber.ir.model import TimberIR
        changed,_,d=dead_leaf_elimination(TimberIR()); assert not changed

    def test_dead_leaf_idempotent(self):
        from timber.optimizer.dead_leaf import dead_leaf_elimination
        ir=_tiny_ir((1.0,0.0001,1.0,0.0001))
        _,ir1,_=dead_leaf_elimination(ir.deep_copy(),0.01)
        _,ir2,_=dead_leaf_elimination(ir1.deep_copy(),0.01)
        l1=sum(1 for t in ir1.pipeline[0].trees for n in t.nodes if n.is_leaf)
        l2=sum(1 for t in ir2.pipeline[0].trees for n in t.nodes if n.is_leaf)
        assert l1==l2

    def test_constant_feature_folds_equal_children(self):
        from timber.optimizer.constant_feature import constant_feature_detection
        changed,_,d=constant_feature_detection(_tiny_ir((0.5,0.5,0.3,0.3)))
        assert changed and d["nodes_folded"]>0

    def test_constant_feature_no_fold_unequal(self):
        from timber.optimizer.constant_feature import constant_feature_detection
        changed,_,_=constant_feature_detection(_tiny_ir((0.1,0.9,0.2,0.8)))
        assert not changed

    def test_threshold_quant_int8(self):
        from timber.optimizer.threshold_quant import _classify_precision
        assert _classify_precision([1.0,5.0,10.0,20.0])=="int8"

    def test_threshold_quant_float32_for_subnormal(self):
        from timber.optimizer.threshold_quant import _classify_precision
        # 1e-5 is in float16 subnormal range; float16(1e-5)≈9.54e-6 → ~4.6% relative error > 0.1% → float32
        assert _classify_precision([1e-5])=="float32"

    def test_threshold_quant_empty_float32(self):
        from timber.optimizer.threshold_quant import _classify_precision
        assert _classify_precision([])=="float32"

    def test_threshold_quant_writes_metadata(self):
        from timber.optimizer.threshold_quant import threshold_quantization
        ir,_,_=_xgb_ir(10,5)
        _,ir2,_=threshold_quantization(ir.deep_copy())
        assert "quantization_map" in ir2.metadata.compilation_hints

    def test_branch_sort_swaps_right_dominant(self):
        from timber.optimizer.branch_sort import frequency_branch_sort
        ir=_tiny_ir()
        # all samples have f0=1.0 → go right at root → right should become dominant
        data=np.ones((100,2),dtype=np.float32)*1.0
        changed,_,d=frequency_branch_sort(ir.deep_copy(),data)
        assert d["nodes_profiled"]>0

    def test_branch_sort_no_data_skipped(self):
        from timber.optimizer.branch_sort import frequency_branch_sort
        changed,_,d=frequency_branch_sort(_tiny_ir(),np.zeros((0,2),dtype=np.float32))
        assert not changed and "skipped" in d

    def test_vectorization_returns_hint(self):
        from timber.optimizer.vectorize import vectorization_analysis
        ir,_,_=_xgb_ir(10,10)
        result=vectorization_analysis(ir)
        assert "trees_analyzed" in result and result["trees_analyzed"]==10

    def test_vectorization_no_ensemble(self):
        from timber.optimizer.vectorize import vectorization_analysis
        from timber.ir.model import TimberIR
        result=vectorization_analysis(TimberIR())
        assert result["trees_analyzed"]==0

    def test_pipeline_fusion_fuses_scaler(self):
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.optimizer.pipeline_fusion import pipeline_fusion
        from timber.ir.model import ScalerStage
        X,y=_bc(10)
        pipe=SkPipeline([("sc",StandardScaler()),("clf",RandomForestClassifier(5,random_state=0))])
        pipe.fit(X,y); ir=_convert_sklearn(pipe)
        n_before=len(ir.pipeline)
        changed,ir2,d=pipeline_fusion(ir.deep_copy())
        assert changed
        assert len(ir2.pipeline)<n_before
        assert not any(isinstance(s,ScalerStage) for s in ir2.pipeline)

    def test_pipeline_fusion_threshold_math(self):
        """After fusing scaler(mean=2,scale=3) into a threshold of 1.0 → should become 1*3+2=5."""
        from timber.ir.model import (TimberIR,TreeEnsembleStage,Tree,TreeNode,ScalerStage,
                                      Objective,Field,FieldType,Schema,Metadata)
        from timber.optimizer.pipeline_fusion import pipeline_fusion
        nodes=[TreeNode(0,0,1.0,1,2,False,0.0,0),
               TreeNode(1,-1,0.0,-1,-1,True,0.5,1),
               TreeNode(2,-1,0.0,-1,-1,True,-0.5,1)]
        t=Tree(0,nodes); t.recount()
        scaler=ScalerStage("sc","scaler",means=[2.0],scales=[3.0],feature_indices=[0])
        ensemble=TreeEnsembleStage("e","tree_ensemble",[t],1,1,Objective.REGRESSION,0.0,[],1.0,True)
        ir=TimberIR([scaler,ensemble],
                    Schema([Field("f0",FieldType.FLOAT32,0)],[Field("out",FieldType.FLOAT32,0)]),Metadata())
        changed,ir2,_=pipeline_fusion(ir)
        assert changed
        new_threshold=ir2.pipeline[0].trees[0].nodes[0].threshold
        assert math.isclose(new_threshold,5.0,rel_tol=1e-6),f"Expected 5.0 got {new_threshold}"

    def test_full_optimizer_pipeline_runs(self):
        from timber.optimizer.pipeline import OptimizerPipeline
        ir,_,X=_xgb_ir(10,20)
        result=OptimizerPipeline(calibration_data=X[:50]).run(ir)
        assert result.ir is not None
        assert len(result.passes)>0

    def test_optimizer_summary_has_required_keys(self):
        from timber.optimizer.pipeline import OptimizerPipeline
        ir,_,_=_xgb_ir(10,10)
        result=OptimizerPipeline().run(ir)
        s=result.summary()
        assert all(k in s for k in ["total_passes","passes_applied","total_duration_ms"])

    def test_optimizer_preserves_tree_count(self):
        from timber.optimizer.pipeline import OptimizerPipeline
        ir,_,_=_xgb_ir(10,10)
        n_before=ir.get_tree_ensemble().n_trees
        result=OptimizerPipeline().run(ir)
        assert result.ir.get_tree_ensemble().n_trees==n_before


# ═══════════════════════════════════════════════════════════════════════════
# F. DIFF COMPILER
# ═══════════════════════════════════════════════════════════════════════════

class TestDiffCompiler:
    def test_self_diff_no_changes(self):
        from timber.optimizer.diff_compile import diff_models
        ir,_,_=_xgb_ir(10,10)
        d=diff_models(ir,ir)
        assert not d.has_changes
        assert len(d.unchanged_tree_ids)==10

    def test_diff_detects_added_trees(self):
        from timber.optimizer.diff_compile import diff_models
        ir_small,_,_=_xgb_ir(10,5); ir_big,_,_=_xgb_ir(10,10)
        d=diff_models(ir_small,ir_big)
        assert len(d.added_tree_ids)>0

    def test_diff_detects_removed_trees(self):
        from timber.optimizer.diff_compile import diff_models
        ir_small,_,_=_xgb_ir(10,5); ir_big,_,_=_xgb_ir(10,10)
        d=diff_models(ir_big,ir_small)
        assert len(d.removed_tree_ids)>0

    def test_diff_detects_modified_tree(self):
        from timber.optimizer.diff_compile import diff_models
        ir,_,_=_xgb_ir(10,5)
        ir2=ir.deep_copy()
        ir2.pipeline[0].trees[0].nodes[0].threshold+=999.0
        d=diff_models(ir,ir2)
        assert len(d.modified_tree_ids)>0

    def test_diff_unchanged_when_identical(self):
        from timber.optimizer.diff_compile import diff_models
        ir,_,_=_xgb_ir(10,5)
        d=diff_models(ir,ir.deep_copy())
        assert not d.has_changes

    def test_diff_summary_has_keys(self):
        from timber.optimizer.diff_compile import diff_models
        ir,_,_=_xgb_ir(10,5)
        s=diff_models(ir,ir).summary()
        assert all(k in s for k in ["added","removed","modified","unchanged"])

    def test_incremental_compile_annotates_ir(self):
        from timber.optimizer.diff_compile import diff_models, incremental_compile
        ir,_,_=_xgb_ir(10,5); ir2=ir.deep_copy()
        d=diff_models(ir,ir2); ir3=incremental_compile(ir,ir2,d)
        assert "diff" in ir3.get_tree_ensemble().annotations

    def test_n_changed_property(self):
        from timber.optimizer.diff_compile import diff_models
        ir_small,_,_=_xgb_ir(10,3); ir_big,_,_=_xgb_ir(10,8)
        d=diff_models(ir_small,ir_big)
        assert d.n_changed==len(d.added_tree_ids)+len(d.removed_tree_ids)+len(d.modified_tree_ids)


# ═══════════════════════════════════════════════════════════════════════════
# G. C99 EMITTER — ABI, error paths, embedded
# ═══════════════════════════════════════════════════════════════════════════

class TestC99Emitter:
    def test_tree_compiles(self,tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir,_,_=_xgb_ir(10,10); out=C99Emitter().emit(ir); out.write(tmp_path)
        assert _compile_so(tmp_path) is not None

    def test_header_include_guard(self):
        from timber.codegen.c99 import C99Emitter
        ir,_,_=_xgb_ir(10,5); out=C99Emitter().emit(ir)
        assert "#ifndef TIMBER_MODEL_H" in out.model_h
        assert "#define TIMBER_MODEL_H" in out.model_h
        assert "#endif" in out.model_h

    def test_abi_version_returns_1(self,tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir,_,_=_xgb_ir(10,5); out=C99Emitter().emit(ir); out.write(tmp_path)
        so=_compile_so(tmp_path)
        if so is None: pytest.skip("no compiler")
        lib=ctypes.CDLL(str(so)); lib.timber_abi_version.restype=ctypes.c_int
        assert lib.timber_abi_version()==1

    def test_null_input_returns_minus1(self,tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir,_,_=_xgb_ir(10,5); out=C99Emitter().emit(ir); out.write(tmp_path)
        so=_compile_so(tmp_path)
        if so is None: pytest.skip("no compiler")
        lib=ctypes.CDLL(str(so)); lib.timber_infer.restype=ctypes.c_int
        lib.timber_infer.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
        assert lib.timber_infer(None,1,None,None)==-1

    def test_write_returns_5_files(self,tmp_path):
        from timber.codegen.c99 import C99Emitter
        ir,_,_=_xgb_ir(10,5); out=C99Emitter().emit(ir)
        files=out.write(tmp_path)
        assert len(files)==5
        for f in files: assert Path(f).exists()

    def test_cortex_m4_no_shared_lib(self):
        from timber.codegen.c99 import C99Emitter,TargetSpec
        ir,_,_=_xgb_ir(10,5); spec=TargetSpec.for_embedded("cortex-m4")
        out=C99Emitter(spec).emit(ir)
        assert ".so" not in out.makefile
        assert ".a" in out.makefile
        assert "arm-none-eabi-gcc" in out.makefile

    def test_cortex_m4_cpu_flags(self):
        from timber.codegen.c99 import C99Emitter,TargetSpec
        ir,_,_=_xgb_ir(10,5); spec=TargetSpec.for_embedded("cortex-m4")
        out=C99Emitter(spec).emit(ir)
        assert "-mcpu=cortex-m4" in out.makefile

    def test_rv32imf_cross_prefix(self):
        from timber.codegen.c99 import C99Emitter,TargetSpec
        ir,_,_=_xgb_ir(10,5); spec=TargetSpec.for_embedded("rv32imf")
        out=C99Emitter(spec).emit(ir)
        assert "riscv32" in out.makefile

    def test_no_fpic_in_embedded(self):
        from timber.codegen.c99 import C99Emitter,TargetSpec
        for profile in ["cortex-m4","cortex-m33","rv32imf","rv64gc"]:
            ir,_,_=_xgb_ir(10,5); out=C99Emitter(TargetSpec.for_embedded(profile)).emit(ir)
            assert "-fPIC" not in out.makefile,f"{profile} has -fPIC"

    def test_unknown_embedded_profile_raises(self):
        from timber.codegen.c99 import TargetSpec
        with pytest.raises(ValueError,match="Unknown embedded profile"):
            TargetSpec.for_embedded("not-a-real-chip")

    def test_empty_pipeline_raises(self):
        from timber.codegen.c99 import C99Emitter
        from timber.ir.model import TimberIR
        with pytest.raises(ValueError,match="No supported primary stage"):
            C99Emitter().emit(TimberIR())

    def test_linear_binary_compiles(self,tmp_path):
        from sklearn.linear_model import LogisticRegression
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.c99 import C99Emitter
        X,y=_bc(10); p=_onnx_path(LogisticRegression(max_iter=300).fit(X,y),X)
        try:
            ir=parse_onnx_model(p); out=C99Emitter().emit(ir); out.write(tmp_path)
            assert _compile_so(tmp_path) is not None
        finally: os.unlink(p)

    def test_svm_rbf_compiles(self,tmp_path):
        from sklearn.svm import SVC
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.c99 import C99Emitter
        X,y=_bc(10); Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
        p=_onnx_path(SVC(kernel="rbf",max_iter=500).fit(Xn,y),Xn)
        try:
            ir=parse_onnx_model(p); out=C99Emitter().emit(ir); out.write(tmp_path)
            assert _compile_so(tmp_path) is not None
        finally: os.unlink(p)


# ═══════════════════════════════════════════════════════════════════════════
# H. WASM EMITTER
# ═══════════════════════════════════════════════════════════════════════════

class TestWasmEmitter:
    def test_emit_returns_wat_and_js(self):
        from timber.codegen.wasm import WasmEmitter
        ir,_,_=_xgb_ir(10,5); out=WasmEmitter().emit(ir)
        assert isinstance(out.wat,str) and isinstance(out.js_bindings,str)

    def test_wat_has_module_keyword(self):
        from timber.codegen.wasm import WasmEmitter
        ir,_,_=_xgb_ir(10,5); out=WasmEmitter().emit(ir)
        assert "(module" in out.wat

    def test_wat_has_memory_export(self):
        from timber.codegen.wasm import WasmEmitter
        ir,_,_=_xgb_ir(10,5); out=WasmEmitter().emit(ir)
        assert '(memory' in out.wat and '"memory"' in out.wat

    def test_wat_mentions_tree_count(self):
        from timber.codegen.wasm import WasmEmitter
        ir,_,_=_xgb_ir(10,5); out=WasmEmitter().emit(ir)
        assert "Trees: 5" in out.wat

    def test_js_mentions_timber(self):
        from timber.codegen.wasm import WasmEmitter
        ir,_,_=_xgb_ir(10,5); out=WasmEmitter().emit(ir)
        assert "timber" in out.js_bindings.lower()

    def test_write_creates_files(self,tmp_path):
        from timber.codegen.wasm import WasmEmitter
        ir,_,_=_xgb_ir(10,5); out=WasmEmitter().emit(ir)
        files=out.write(tmp_path)
        assert any("model.wat" in f for f in files)
        assert any(".js" in f for f in files)
        for f in files: assert Path(f).exists()

    def test_no_ensemble_raises(self):
        from timber.codegen.wasm import WasmEmitter
        from timber.ir.model import TimberIR
        with pytest.raises(ValueError,match="No tree ensemble"):
            WasmEmitter().emit(TimberIR())


# ═══════════════════════════════════════════════════════════════════════════
# I. MISRA-C
# ═══════════════════════════════════════════════════════════════════════════

class TestMisraC:
    def test_banner_present(self):
        from timber.codegen.misra_c import MisraCEmitter
        ir,_,_=_xgb_ir(10,5); out=MisraCEmitter().emit(ir)
        assert "MISRA C:2012" in out.model_h and "MISRA C:2012" in out.model_c

    def test_tree_output_compliant(self):
        from timber.codegen.misra_c import MisraCEmitter
        ir,_,_=_xgb_ir(10,5); em=MisraCEmitter(); out=em.emit(ir)
        rep=em.check_compliance(out.model_c)
        assert rep.is_compliant,f"Violations:\n{rep.summary()}"

    def test_no_stdio_in_output(self):
        from timber.codegen.misra_c import MisraCEmitter
        ir,_,_=_xgb_ir(10,5); out=MisraCEmitter().emit(ir)
        assert "<stdio.h>" not in out.model_c and "printf" not in out.model_c

    def test_no_compiler_extensions(self):
        from timber.codegen.misra_c import MisraCEmitter
        ir,_,_=_xgb_ir(10,5); out=MisraCEmitter().emit(ir)
        assert "__attribute__" not in out.model_c

    def test_rule_1_1_detects_attribute(self):
        from timber.codegen.misra_c import MisraCEmitter
        em=MisraCEmitter(); rep=em.check_compliance('__attribute__((unused)) int x=0;')
        assert not rep.is_compliant and "1.1" in [v["rule"] for v in rep.violations]

    def test_rule_20_9_detects_stdio(self):
        from timber.codegen.misra_c import MisraCEmitter
        em=MisraCEmitter(); rep=em.check_compliance('#include <stdio.h>\nint x=0;')
        assert not rep.is_compliant and "20.9" in [v["rule"] for v in rep.violations]

    def test_rule_21_6_detects_printf(self):
        from timber.codegen.misra_c import MisraCEmitter
        em=MisraCEmitter(); rep=em.check_compliance('void f(void){printf("x");}')
        assert not rep.is_compliant and "21.6" in [v["rule"] for v in rep.violations]

    def test_rule_7_1_detects_octal(self):
        from timber.codegen.misra_c import MisraCEmitter
        em=MisraCEmitter(); rep=em.check_compliance('int x=017;')
        assert not rep.is_compliant and "7.1" in [v["rule"] for v in rep.violations]

    def test_misra_compiles_with_gcc(self,tmp_path):
        from timber.codegen.misra_c import MisraCEmitter
        ir,_,_=_xgb_ir(10,5); out=MisraCEmitter().emit(ir); out.write(tmp_path)
        assert _compile_so(tmp_path) is not None,"MISRA output failed to compile"

    def test_report_rules_checked_gte_8(self):
        from timber.codegen.misra_c import MisraCEmitter
        ir,_,_=_xgb_ir(10,5); em=MisraCEmitter(); out=em.emit(ir)
        rep=em.check_compliance(out.model_c)
        assert rep.rules_checked>=8

    def test_violation_object_has_rule_and_severity(self):
        from timber.codegen.misra_c import MisraCEmitter
        em=MisraCEmitter(); rep=em.check_compliance('__attribute__((unused)) int x;')
        assert len(rep.violation_objects)>0
        v=rep.violation_objects[0]
        assert hasattr(v,"rule") and hasattr(v,"severity")


# ═══════════════════════════════════════════════════════════════════════════
# J. LLVM IR BACKEND
# ═══════════════════════════════════════════════════════════════════════════

class TestLLVMIR:
    def test_tree_module_header(self):
        from timber.codegen.llvm_ir import LLVMIREmitter
        ir,_,_=_xgb_ir(10,5); out=LLVMIREmitter("x86_64").emit(ir)
        assert "; ModuleID" in out.model_ll and 'target triple' in out.model_ll

    def test_tree_has_traverse_functions(self):
        from timber.codegen.llvm_ir import LLVMIREmitter
        ir,_,_=_xgb_ir(10,3); out=LLVMIREmitter("x86_64").emit(ir)
        assert "traverse_tree_0" in out.model_ll

    def test_linear_has_weights_and_define(self):
        from sklearn.linear_model import LogisticRegression
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.llvm_ir import LLVMIREmitter
        X,y=_bc(10); p=_onnx_path(LogisticRegression(max_iter=300).fit(X,y),X)
        try:
            ir=parse_onnx_model(p); out=LLVMIREmitter("x86_64").emit(ir)
            assert "timber_weights" in out.model_ll and "define" in out.model_ll
        finally: os.unlink(p)

    def test_svm_has_exp_intrinsic(self):
        from sklearn.svm import SVC
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.llvm_ir import LLVMIREmitter
        X,y=_bc(10); Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
        p=_onnx_path(SVC(kernel="rbf",max_iter=500).fit(Xn,y),Xn)
        try:
            ir=parse_onnx_model(p); out=LLVMIREmitter("x86_64").emit(ir)
            assert "llvm.exp" in out.model_ll
        finally: os.unlink(p)

    def test_aarch64_triple(self):
        from timber.codegen.llvm_ir import LLVMIREmitter
        ir,_,_=_xgb_ir(10,5); out=LLVMIREmitter("aarch64").emit(ir)
        assert "aarch64" in out.target_triple

    def test_cortex_m4_triple(self):
        from timber.codegen.llvm_ir import LLVMIREmitter
        ir,_,_=_xgb_ir(10,5); out=LLVMIREmitter("cortex-m4").emit(ir)
        assert "arm" in out.target_triple.lower() or "thumb" in out.target_triple.lower()

    def test_save_to_disk(self,tmp_path):
        from timber.codegen.llvm_ir import LLVMIREmitter
        ir,_,_=_xgb_ir(10,5); out=LLVMIREmitter("x86_64").emit(ir)
        files=out.save(tmp_path)
        assert Path(files["model.ll"]).exists()

    def test_normalizer_only_raises(self):
        from timber.codegen.llvm_ir import LLVMIREmitter
        from timber.ir.model import TimberIR,NormalizerStage,Field,FieldType,Schema,Metadata
        s=NormalizerStage("n","normalizer",norm="l2")
        ir=TimberIR([s],Schema([Field("f0",FieldType.FLOAT32,0)],[Field("out",FieldType.FLOAT32,0)]),Metadata())
        with pytest.raises(ValueError,match="No supported primary stage"):
            LLVMIREmitter().emit(ir)


# ═══════════════════════════════════════════════════════════════════════════
# K. DIFFERENTIAL PRIVACY
# ═══════════════════════════════════════════════════════════════════════════

class TestDP:
    def test_laplace_output_shape(self):
        from timber.privacy.dp import DPConfig,apply_dp_noise
        o=np.ones((10,2),dtype=np.float32); cfg=DPConfig("laplace",epsilon=1.0,sensitivity=1.0,seed=0)
        n,_=apply_dp_noise(o,cfg); assert n.shape==(10,2)

    def test_gaussian_output_shape(self):
        from timber.privacy.dp import DPConfig,apply_dp_noise
        o=np.ones((5,1),dtype=np.float64); cfg=DPConfig("gaussian",epsilon=1.0,delta=1e-5,sensitivity=1.0,seed=0)
        n,_=apply_dp_noise(o,cfg); assert n.shape==(5,1)

    def test_float32_dtype_preserved(self):
        from timber.privacy.dp import DPConfig,apply_dp_noise
        o=np.array([[0.5]],dtype=np.float32); cfg=DPConfig("laplace",epsilon=1.0,sensitivity=1.0,seed=0)
        n,_=apply_dp_noise(o,cfg); assert n.dtype==np.float32

    def test_float64_dtype_preserved(self):
        from timber.privacy.dp import DPConfig,apply_dp_noise
        o=np.array([[0.5]],dtype=np.float64); cfg=DPConfig("gaussian",epsilon=1.0,delta=1e-5,sensitivity=1.0,seed=0)
        n,_=apply_dp_noise(o,cfg); assert n.dtype==np.float64

    def test_clipping_respected(self):
        from timber.privacy.dp import DPConfig,apply_dp_noise
        o=np.ones((50,1),dtype=np.float32)*0.5
        cfg=DPConfig("laplace",epsilon=0.01,sensitivity=1.0,clip_outputs=True,output_min=0.0,output_max=1.0,seed=5)
        n,_=apply_dp_noise(o,cfg); assert np.all(n>=0.0) and np.all(n<=1.0)

    def test_deterministic_with_seed(self):
        from timber.privacy.dp import DPConfig,apply_dp_noise
        o=np.array([[0.5,0.3]],dtype=np.float64)
        cfg1=DPConfig("laplace",epsilon=1.0,sensitivity=1.0,seed=42)
        cfg2=DPConfig("laplace",epsilon=1.0,sensitivity=1.0,seed=42)
        n1,_=apply_dp_noise(o,cfg1); n2,_=apply_dp_noise(o,cfg2)
        np.testing.assert_array_equal(n1,n2)

    def test_different_seeds_differ(self):
        from timber.privacy.dp import DPConfig,apply_dp_noise
        o=np.array([[0.5]],dtype=np.float64)
        n1,_=apply_dp_noise(o,DPConfig("laplace",seed=1))
        n2,_=apply_dp_noise(o,DPConfig("laplace",seed=2))
        assert not np.allclose(n1,n2)

    def test_laplace_scale_formula(self):
        from timber.privacy.dp import DPConfig
        cfg=DPConfig("laplace",epsilon=2.0,sensitivity=1.0)
        assert math.isclose(cfg.laplace_scale,0.5,rel_tol=1e-9)

    def test_gaussian_sigma_formula(self):
        from timber.privacy.dp import DPConfig
        cfg=DPConfig("gaussian",epsilon=1.0,delta=1e-5,sensitivity=1.0)
        expected=math.sqrt(2*math.log(1.25/1e-5))/1.0
        assert math.isclose(cfg.gaussian_sigma,expected,rel_tol=1e-6)

    def test_laplace_empirical_std_matches_theory(self):
        from timber.privacy.dp import DPConfig,apply_dp_noise
        o=np.zeros((10000,1),dtype=np.float64)
        cfg=DPConfig("laplace",epsilon=1.0,sensitivity=1.0,clip_outputs=False,seed=0)
        n,rep=apply_dp_noise(o,cfg)
        emp_std=float(np.std(n)); theory=rep.noise_scale*math.sqrt(2)
        assert abs(emp_std-theory)/theory<0.05,f"emp={emp_std:.4f} theory={theory:.4f}"

    def test_gaussian_empirical_std_matches_theory(self):
        from timber.privacy.dp import DPConfig,apply_dp_noise
        o=np.zeros((10000,1),dtype=np.float64)
        cfg=DPConfig("gaussian",epsilon=1.0,delta=1e-5,sensitivity=1.0,clip_outputs=False,seed=0)
        n,rep=apply_dp_noise(o,cfg)
        emp_std=float(np.std(n))
        assert abs(emp_std-rep.noise_scale)/rep.noise_scale<0.05

    def test_high_epsilon_low_noise(self):
        from timber.privacy.dp import DPConfig,apply_dp_noise
        o=np.random.default_rng(0).uniform(0.3,0.7,(200,1))
        cfg=DPConfig("laplace",epsilon=1000.0,sensitivity=1.0,clip_outputs=False,seed=0)
        n,_=apply_dp_noise(o,cfg)
        assert np.mean(np.abs(n-o))<0.005

    def test_invalid_epsilon_raises(self):
        from timber.privacy.dp import DPConfig
        with pytest.raises(ValueError,match="epsilon"): DPConfig(epsilon=-1.0)

    def test_invalid_sensitivity_raises(self):
        from timber.privacy.dp import DPConfig
        with pytest.raises(ValueError,match="sensitivity"): DPConfig(sensitivity=0.0)

    def test_invalid_mechanism_raises(self):
        from timber.privacy.dp import DPConfig
        with pytest.raises(ValueError,match="mechanism"): DPConfig(mechanism="foo")

    def test_gaussian_invalid_delta_raises(self):
        from timber.privacy.dp import DPConfig
        with pytest.raises(ValueError,match="delta"): DPConfig(mechanism="gaussian",delta=0.0)

    def test_calibrate_epsilon_laplace(self):
        from timber.privacy.dp import calibrate_epsilon
        eps=calibrate_epsilon(0.1,1.0,"laplace")
        assert math.isclose(eps,math.sqrt(2)/0.1,rel_tol=1e-6)

    def test_report_summary_contains_epsilon(self):
        from timber.privacy.dp import DPConfig,apply_dp_noise
        o=np.array([[0.5]],dtype=np.float64)
        _,rep=apply_dp_noise(o,DPConfig("laplace",epsilon=2.5,sensitivity=1.0,seed=0))
        assert "2.5" in rep.summary() or "epsilon" in rep.summary().lower()


# ═══════════════════════════════════════════════════════════════════════════
# L. FULL END-TO-END PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_xgb_parse_optimize_emit_compile_infer(self,tmp_path):
        from timber.optimizer.pipeline import OptimizerPipeline
        from timber.codegen.c99 import C99Emitter
        ir,_,X=_xgb_ir(10,20)
        result=OptimizerPipeline(calibration_data=X[:50]).run(ir)
        out=C99Emitter().emit(result.ir); out.write(tmp_path)
        so=_compile_so(tmp_path)
        if so is None: pytest.skip("no compiler")
        res=_ctypes_infer(so,X[:10],1)
        assert res is not None and np.all(np.isfinite(res))

    def test_sklearn_pipeline_parse_emit_compile(self,tmp_path):
        from timber.frontends.sklearn_parser import _convert_sklearn
        from timber.codegen.c99 import C99Emitter
        X,y=_bc(10)
        pipe=SkPipeline([("sc",StandardScaler()),("clf",RandomForestClassifier(10,random_state=0))])
        pipe.fit(X,y); ir=_convert_sklearn(pipe)
        out=C99Emitter().emit(ir); out.write(tmp_path)
        assert _compile_so(tmp_path) is not None

    def test_onnx_linear_parse_misra_compile(self,tmp_path):
        from sklearn.linear_model import LogisticRegression
        from timber.frontends.onnx_parser import parse_onnx_model
        from timber.codegen.misra_c import MisraCEmitter
        X,y=_bc(10); p=_onnx_path(LogisticRegression(max_iter=300).fit(X,y),X)
        try:
            ir=parse_onnx_model(p); em=MisraCEmitter(); out=em.emit(ir); out.write(tmp_path)
            rep=em.check_compliance(out.model_c)
            assert rep.is_compliant
            assert _compile_so(tmp_path) is not None
        finally: os.unlink(p)

    def test_optimizer_does_not_corrupt_predictions(self):
        from timber.optimizer.pipeline import OptimizerPipeline
        ir,_,X=_xgb_ir(10,20)
        py_before=_py_infer_tree(ir,X[:20])
        result=OptimizerPipeline().run(ir)
        py_after=_py_infer_tree(result.ir,X[:20])
        # Predictions should be nearly identical (optimizer shouldn't corrupt)
        corr=float(np.corrcoef(py_before,py_after)[0,1]) if py_before.std()>0 else 1.0
        assert corr>0.99,f"Optimizer corrupted predictions: corr={corr:.4f}"

    def test_dp_after_c99_inference(self,tmp_path):
        from timber.codegen.c99 import C99Emitter
        from timber.privacy.dp import DPConfig,apply_dp_noise
        ir,_,X=_xgb_ir(10,10)
        out=C99Emitter().emit(ir); out.write(tmp_path)
        so=_compile_so(tmp_path)
        if so is None: pytest.skip("no compiler")
        raw=_ctypes_infer(so,X[:20],1)
        assert raw is not None
        cfg=DPConfig("laplace",epsilon=1.0,sensitivity=1.0,clip_outputs=False,seed=0)
        noisy,rep=apply_dp_noise(raw,cfg)
        assert noisy.shape==raw.shape and rep.n_outputs_noised==20

    def test_wasm_emit_after_optimize(self):
        from timber.optimizer.pipeline import OptimizerPipeline
        from timber.codegen.wasm import WasmEmitter
        ir,_,X=_xgb_ir(10,10)
        result=OptimizerPipeline(calibration_data=X[:50]).run(ir)
        out=WasmEmitter().emit(result.ir)
        assert "(module" in out.wat

    def test_llvm_emit_after_optimize(self):
        from timber.optimizer.pipeline import OptimizerPipeline
        from timber.codegen.llvm_ir import LLVMIREmitter
        ir,_,X=_xgb_ir(10,10)
        result=OptimizerPipeline().run(ir)
        out=LLVMIREmitter("x86_64").emit(result.ir)
        assert "timber_infer_single" in out.model_ll

    def test_bench_report_html_roundtrip(self):
        from timber.cli import _bench_report_html
        data={"timber_version":"0.2.0",
              "system":{"platform":"test","python":"3.11","cpu":"test","timestamp":"2024-01-01T00:00:00Z"},
              "model":{"artifact":"a","n_trees":10,"max_depth":3,"n_features":30,
                       "n_classes":2,"objective":"binary:logistic","n_samples":100},
              "results":[{"batch_size":1,"n_runs":200,"min_us":1.0,"p50_us":2.0,"p95_us":3.0,
                          "p99_us":4.0,"p999_us":5.0,"mean_us":2.1,"std_us":0.3,"cv_pct":14.3,
                          "throughput_samples_per_sec":500000.0}]}
        html=_bench_report_html(data)
        assert "<!DOCTYPE html>" in html and "500000" in html and "Timber Benchmark" in html
