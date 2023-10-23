import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.quantization import QuantStub, DeQuantStub

def quantize_model(model):
    # Agregar los módulos de QuantStub y DeQuantStub al modelo
    model.quant = QuantStub()
    model.dequant = DeQuantStub()

    # Marcar el modelo como listo para la cuantización
    model.qconfig = torch.quantization.default_qconfig
    model = torch.quantization.prepare(model, inplace=True)

    # Calibrar la cuantización
    calibration_data = "Ejemplo de texto para calibrar la cuantización."
    input_ids = tokenizer.encode(calibration_data, return_tensors='pt')
    with torch.no_grad():
        model(input_ids)
    
    # Aplicar la cuantización
    model = torch.quantization.convert(model, inplace=True)
    
    return model

# Cargar el modelo GPT-2 y el tokenizador
model_id = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Cuantizar el modelo
quantized_model = quantize_model(model)

# Generar texto con el modelo cuantizado
input_text = "I have a dream"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = quantized_model.generate(input_ids, max_length=50, do_sample=True, top_k=30, pad_token_id=tokenizer.eos_token_id, attention_mask=input_ids.new_ones(input_ids.shape))
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Texto generado con el modelo cuantizado:\n{generated_text}")
