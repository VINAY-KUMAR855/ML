import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model(
    "best_resnet50_finetuned.keras",
    compile=False
)

input_signature = [
    tf.TensorSpec(
        shape=(None, *model.input_shape[1:]),
        dtype=tf.float32,
        name="input"
    )
]

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=15,
    output_path="best_resnet50_finetuned.onnx"
)

print("Conversion completed!")
