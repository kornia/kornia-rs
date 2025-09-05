# !pip install onnx-graphsurgeon

import onnx
import onnx_graphsurgeon as gs
import torch, random, numpy as np
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers import set_seed
from safetensors.torch import save_file as save_safetensor


DEVICE = "cuda"

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
set_all_seeds(0)


def onnx_extract_layer():
    input_onnx_path = "../validation_data/decoder_model_merged.onnx"
    output_onnx_path = "../validation_data/extracted_layer.onnx"

    graph = gs.import_onnx(onnx.load(input_onnx_path))

    print("All node names in the model:")
    for node in graph.nodes:
        print(node.name, node.op)

    # --- USER: Set this to the name of the layer you want to extract ---
    target_node_name = "/model/layers.0/attn/q_proj/MatMul"

    # Find the node
    target_node = None
    for node in graph.nodes:
        if node.name == target_node_name:
            target_node = node
            break

    if target_node is None:
        raise ValueError(f"Node {target_node_name} not found in the model.")

    print(">>>", target_node.outputs)
    print("<<<", target_node.inputs)

    # Identify the activation input (not weights/biases)
    activation_input = None
    for inp in target_node.inputs:
        if inp.name in [i.name for i in graph.inputs]:
            activation_input = inp
            break
    if activation_input is None:
        # Then ssume the first input is the activation
        activation_input = target_node.inputs[0]

    graph.inputs = [activation_input]
    graph.outputs = target_node.outputs

    graph.cleanup()
    onnx.save(gs.export_onnx(graph), output_onnx_path)

    print(f"Extracted layer saved to {output_onnx_path}")

# Specify the layer index you want to save activations for
def generate_inp_out_for_onnx_single_layer_validation(
    layer_idx = 0,  # Change this to the desired layer index
    token_pos = 10  # Change this to the desired token position if needed
):
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    messages = [
        {
            "role": "user",
            "content": [
                # {"type": "image"},
                # {"type": "text", "text": "Can you describe the image?"}
                {"type": "text", "text": "What is life?"}
                # {"type": "text", "text": "A real-valued function f defined on the real line is called an even function if f(-t) = f(t) for each real number t. Prove that the set of even functions defined on the real line with the operations of addition and scalar multiplication defined in Example 3 is a vector space."}
            ]
        },
    ]


    # Initialize model directly on CUDA without Flash Attention
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.float32,
        # _attn_implementation="flash_attention_2",  # Commented out Flash Attention
        device_map="cpu",
    )
    model.eval()


    embeddings = {}
    counter_token_pos = -1

    def hook_fn(name, initial_layer: bool = False):
        def hook(module, input, output):
            global counter_token_pos
            if initial_layer:
                counter_token_pos += 1
            
            if isinstance(output, tuple):
                output = output[0]
            elif isinstance(output, torch.Tensor):
                pass
            else:
                print("Hook unknown type!!!", name, type(output))

            embeddings[name+f"_{counter_token_pos}"] = output.detach().cpu()

        return hook

    # Register hooks for different layers
    model.get_input_embeddings().register_forward_hook(hook_fn("input_embeddings", initial_layer=True))
    for i in range(24):
        model.model.text_model.layers[i].input_layernorm.register_forward_hook(hook_fn(f"input_layernorm_d{i}"))
        model.model.text_model.layers[i].self_attn.register_forward_hook(hook_fn(f"self_attn_d{i}"))
        model.model.text_model.layers[i].self_attn.k_proj.register_forward_hook(hook_fn(f"self_attn_k_proj_d{i}"))
        model.model.text_model.layers[i].self_attn.v_proj.register_forward_hook(hook_fn(f"self_attn_v_proj_d{i}"))
        model.model.text_model.layers[i].self_attn.q_proj.register_forward_hook(hook_fn(f"self_attn_q_proj_d{i}"))
        model.model.text_model.layers[i].self_attn.o_proj.register_forward_hook(hook_fn(f"self_attn_o_proj_d{i}"))
        model.model.text_model.layers[i].post_attention_layernorm.register_forward_hook(hook_fn(f"post_layernorm_d{i}"))
        # model.model.text_model.layers[i].mlp.register_forward_hook(hook_fn(f"mlp_d{i}"))
        model.model.text_model.layers[i].mlp.gate_proj.register_forward_hook(hook_fn(f"mlp_gate_proj_d{i}"))
        model.model.text_model.layers[i].mlp.up_proj.register_forward_hook(hook_fn(f"mlp_up_proj_d{i}"))
        model.model.text_model.layers[i].mlp.down_proj.register_forward_hook(hook_fn(f"mlp_down_proj_d{i}"))
        model.model.text_model.layers[i].mlp.act_fn.register_forward_hook(hook_fn(f"mlp_act_fn_d{i}"))
        model.model.text_model.layers[i].register_forward_hook(hook_fn(f"layers_d{i}"))

    print(type(model))



    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    # inputs = processor(text=prompt, images=[image1], return_tensors="pt")
    inputs = processor(text=prompt, return_tensors="pt")
    inputs = inputs.to("cpu")

    print(inputs["input_ids"])
    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            # repition_penalty=1.1,  # Apply repeat penalty
            output_scores=True,           # Return logits for each generated token
            return_dict_in_generate=True, # Return detailed output object
            do_sample=False,  # Use greedy decoding (highest logit)
        )

    print(outputs.sequences[0])
    print(processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))

    # Keys for input and output activations in the embeddings dict
    input_key = f"input_layernorm_d{layer_idx}_{token_pos}"
    output_key = f"self_attn_q_proj_d{layer_idx}_{token_pos}"

    # Retrieve tensors from embeddings dict
    input_tensor = embeddings.get(input_key)
    output_tensor = embeddings.get(output_key)

    if input_tensor is None or output_tensor is None:
        raise ValueError(f"Could not find input or output tensor for layer {layer_idx} and token {token_pos}.")

    # Prepare dict for safetensors
    to_save = {
        "input": input_tensor,
        "output": output_tensor,
    }

    # Save to safetensors file
    save_path = os.path.join("../validation_data", f"extracted_layer_activations.safetensors")
    save_safetensor(to_save, save_path)
    print(f"Saved activations to {save_path}")


if __name__ == "__main__":
    # onnx_extract_layer()
    # generate_inp_out_for_onnx_single_layer_validation()
    pass
