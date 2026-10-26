import numpy as np
import pytest


def _build_model(path, size=112, channels=1, external=False):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.normal(size=(channels, 7)).astype(np.float32), "W")
    b = numpy_helper.from_array(np.zeros(7, np.float32), "B")
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"]),
        helper.make_node("Gemm", ["flat", "W", "B"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
    ]
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", channels, size, size])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["batch", 7])], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    if external:
        onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                        location=path.name + ".data", size_threshold=0)
    else:
        onnx.save_model(model, str(path))
    return path


@pytest.fixture
def fake_model(tmp_path):
    return _build_model(tmp_path / "emotion_fp_test.onnx")


@pytest.fixture
def fake_model_3ch(tmp_path):
    return _build_model(tmp_path / "three_channel.onnx", channels=3)


@pytest.fixture
def model_missing_data(tmp_path):
    path = _build_model(tmp_path / "broken.onnx", external=True)
    (tmp_path / "broken.onnx.data").unlink()      # tái hiện lỗi: có .onnx nhưng thiếu .onnx.data
    return path
