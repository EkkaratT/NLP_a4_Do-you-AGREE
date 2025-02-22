from flask import Flask, render_template, request
import torch
from sklearn.metrics.pairwise import cosine_similarity
from models.classes import Tokenizer, preprocess_function, BERT, word2id

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_len = 128
max_mask = 5
n_layers = 6    # number of Encoder of Encoder Layer
n_heads  = 12    # number of heads in Multi-Head Attention
d_model  = 768  # Embedding Size 
d_ff = d_model * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2
vocab_size = 6965
max_len = 128

model = BERT(
    n_layers, 
    n_heads, 
    d_model, 
    d_ff, 
    d_k, 
    n_segments, 
    vocab_size, 
    max_len, 
    device
).to(device)
tokenizer = Tokenizer(word2id)

pretrained_weights = torch.load('models/bert_model.pth')
model_dict = model.state_dict()
pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_weights)
model.load_state_dict(model_dict)
model.to(device)

# Function for mean pooling
def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool

# Function to calculate similarity between two sentences
def calculate_similarity(model, tokenizer, sentence_a, sentence_b, device):
    inputs_a = tokenizer.encode([sentence_a])
    inputs_b = tokenizer.encode([sentence_b])

    inputs_ids_a = inputs_a['input_ids'][0].unsqueeze(0).to(device)
    attention_a = inputs_a['attention_mask'][0].unsqueeze(0).to(device)
    inputs_ids_b = inputs_b['input_ids'][0].unsqueeze(0).to(device)
    attention_b = inputs_b['attention_mask'][0].unsqueeze(0).to(device)

    segment_ids = torch.tensor([0] * max_len).unsqueeze(0).repeat(inputs_ids_a.shape[0], 1).to(device)
    masked_pos  = torch.tensor([0] * max_mask).unsqueeze(0).repeat(inputs_ids_a.shape[0], 1).to(device)

    u = model(inputs_ids_a, segment_ids, masked_pos)[2]
    v = model(inputs_ids_b, segment_ids, masked_pos)[2]

    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)

    similarity_score = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]

    return similarity_score

# Function to classify relationship based on similarity score
def classify_nli_relation(similarity_score):
    if similarity_score >= 0.75:
        return "Entailment"
    elif similarity_score >= 0.5:
        return "Neutral"
    else:
        return "Contradiction"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    premise = request.form['premise']
    hypothesis = request.form['hypothesis']
    
    # Compute cosine similarity between premise and hypothesis
    similarity = calculate_similarity(model, tokenizer, premise, hypothesis, device)
    
    # Classify based on the similarity score
    label = classify_nli_relation(similarity)
    
    # Display similarity score and NLI label
    return render_template('index.html', premise=premise, hypothesis=hypothesis, similarity=f"{similarity:.4f}", label=label)

if __name__ == '__main__':
    app.run(debug=True)
