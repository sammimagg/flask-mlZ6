from pathlib import Path
from transformers import XLNetTokenizer, XLNetModel, convert_graph_to_onnx
from transformers import XLNetTokenizerFast
tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased', do_lower_case=True)

model_folder = Path("C:/Users/sammi/Documents/GitHub/DeceptiveTextDetector/model")
tokenizer = XLNetTokenizer.from_pretrained(str(model_folder))
model = XLNetModel.from_pretrained(str(model_folder))

output_path = model_folder / "onnx_model.onnx"

# Use a Path object for the output path
convert_graph_to_onnx.convert("pt", str(model_folder), tokenizer, str(output_path))
