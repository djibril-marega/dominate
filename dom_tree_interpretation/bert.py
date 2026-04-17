from transformers import AutoTokenizer, AutoModel
import re, torch
#from file_manager.json_file_manager import find_key, get_json_data
import torch.nn as nn

def clean_token(text):
    """Simple cleaning for tokens."""
    text = str(text).lower()
    text = re.sub(r"https?://", "", text)        # remove protocole
    text = re.sub(r"[^\w\s]", " ", text)        # remove ponctuation
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    return text.strip()


def serialize_attributs(x, parent_key=None):
    """Recursive serialization of dict, list, or simple values."""
    if isinstance(x, dict):
        parts = []
        for key, value in x.items():
            key_clean = clean_token(key)
            serialized_value = serialize_attributs(value, key_clean)

            if isinstance(value, list):
                parts.append(f"{key_clean} includes {serialized_value}")
            elif isinstance(value, dict):
                parts.append(f"{key_clean} contains {serialized_value}")
            else:
                parts.append(f"{key_clean} is {serialized_value}")

        return " ; ".join(parts)

    elif isinstance(x, list):
        return " and ".join(serialize_attributs(v) for v in x)

    else:
        return clean_token(x)

def serialize_tags(list_tags):
    list_tags_serialized = []
    for tuple_tag in list_tags:
        tag_serialized = (tuple_tag[0], f"tag: {tuple_tag[1]}")
        list_tags_serialized.append(tag_serialized)

    return list_tags_serialized
    


def serialize_tuple(item, serialize_func):
    """
    Take a tuple (id, attrs) and return (id, attrs_serialized)
    """
    element_id, attrs = item

    if attrs is None:
        return (element_id, "")

    try:
        serialized_attrs = serialize_func(attrs)
    except Exception as e:
        print(f"Error while serializing element {element_id}: {attrs}")
        serialized_attrs = ""

    return (element_id, serialized_attrs)

def serialize_tag(item):
    """
    Take a tuple (id, tag) and return (id, tags_serialized)
    """
    element_id, tag = item

    if tag is None:
        return (element_id, "")

    try:
        serialized_tag = f"tag:{tag}"
    except Exception as e:
        print(f"Error while serializing tag {element_id}: {tag}")
        serialized_tag = ""

    return (element_id, serialized_tag)

def generate_mean_pooling_embedding(last_hidden_state, attention_mask=None):
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)
    
    mask = attention_mask.unsqueeze(-1)          # (batch, seq_len, 1)
    masked_embeddings = last_hidden_state #outputs.last_hidden_state * mask  # zero for PADs
    sum_embeddings = masked_embeddings.sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)                         # nb valid tokens
    return sum_embeddings / sum_mask      # mean weighted


def split_text_by_tokens_sentences(text, tokenizer, max_tokens, overlap=1):
    """
    Splits a text into sentence-based chunks with a token limit.
    Allows sliding chunks with overlap.

    Args:
        text (str): text to split
        tokenizer: HuggingFace-compatible tokenizer
        max_tokens (int): maximum number of tokens per chunk
        overlap (int): number of sentences to reuse in the next chunk

    Returns:
        List[str]: list of text chunks
    """
    # split in sentances
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # First step: create chunks without overlap
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(tokenizer(text=sentence)["input_ids"])
        
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            # Start a new chunk with overlap
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk.append(sentence)
            current_tokens = sum(len(tokenizer(text=s)["input_ids"]) for s in current_chunk)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def get_html_list_element_embedding(tokenizer, model, token_max, texts):
    """
    Args : 
    - texts = [(id, text)]
    - token_max = number token max
    """
    list_final_embeddings = []
    for text in texts:
        if text[1] is None: 
            text = (text[0], "EMPTY")
        text_token = len(tokenizer(text=text[1])["input_ids"])
        if text_token > token_max:
            # chunks with overlap
            text_token = len(tokenizer(text=text[1])["input_ids"])
            list_sub_text = split_text_by_tokens_sentences(text, tokenizer, token_max, overlap=2)
            list_sub_text_embedding = []
            for sub_text in list_sub_text:
                tokens_input = tokenizer(text=text[1], return_tensors='pt')
                attention_mask = tokens_input['attention_mask'] 
                outputs = model(**tokens_input)
                sub_text_embedding = generate_mean_pooling_embedding(outputs.last_hidden_state, attention_mask)
                list_sub_text_embedding.append(sub_text_embedding) 

            text_all_sub_embedding = torch.stack(list_sub_text_embedding)
            text_final_embedding = generate_mean_pooling_embedding(text_all_sub_embedding)
        else:
            tokens_input = tokenizer(text=text[1], return_tensors='pt')
            attention_mask = tokens_input['attention_mask'] 
            outputs = model(**tokens_input)
            text_final_embedding = generate_mean_pooling_embedding(outputs.last_hidden_state, attention_mask)
        
        list_final_embeddings.append((text[0], text_final_embedding))
    return list_final_embeddings


class TextualEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        self.token_max = self.model.config.max_position_embeddings
    
    def forward(self, texts, serialized_attributes, serialized_tag):
        texts_embeddings = get_html_list_element_embedding(self.tokenizer, self.model, self.token_max, texts)
        attributs_embeddings = get_html_list_element_embedding(self.tokenizer, self.model, self.token_max, serialized_attributes)
        tags_embeddings = get_html_list_element_embedding(self.tokenizer, self.model, self.token_max, serialized_tag)
        return texts_embeddings, attributs_embeddings, tags_embeddings
