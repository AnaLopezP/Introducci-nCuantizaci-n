import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from codigo import model_id, plt, np, generate_text, calculate_perplexity, original_text, ticker, weights, ppl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 

model_int8 = AutoModelForCausalLM.from_pretrained(model_id,

                                             device_map='auto',

                                             load_in_8bit=True,

                                             )

print(f"Model size: {model_int8.get_memory_footprint():,} bytes")

# Flatten weight tensors

weights_int8 = [param.data.clone() for param in model_int8.parameters()]

weights_int8 = np.concatenate([t.cpu().numpy().flatten() for t in weights_int8])

 

# Set background style

plt.style.use('ggplot')

 

# Create figure and axis

fig, ax = plt.subplots(figsize=(10,5), dpi=300)

 

# Plot the histograms

ax.hist(weights, bins=150, alpha=0.5, label='Original weights',

        color='blue', range=(-2, 2))

ax.hist(weights_int8, bins=150, alpha=0.5, label='LLM.int8() weights',

        color='red', range=(-2, 2))

 

# Add grid

ax.grid(True, linestyle='--', alpha=0.6)

 

# Add legend

ax.legend()

 

# Add title and labels

ax.set_title('Comparison of Original and Dequantized Weights', fontsize=16)

ax.set_xlabel('Weights', fontsize=14)

ax.set_ylabel('Count', fontsize=14)

plt.gca().yaxis.set_major_formatter(ticker.EngFormatter())

 

# Improve font

plt.rc('font', size=12)

 

plt.tight_layout()

plt.show()

# Generate text with quantized model

text_int8 = generate_text(model_int8, "I have a dream")

 

print(f"Original model:\n{original_text}")

print("-" * 50)

print(f"LLM.int8() model:\n{text_int8}")

print(f"Perplexity (original):   {ppl.item():.2f}")

 

ppl = calculate_perplexity(model_int8, text_int8)

print(f"Perplexity (LLM.int8()): {ppl.item():.2f}")