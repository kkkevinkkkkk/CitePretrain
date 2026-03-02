
from transformers import LogitsProcessor, LogitsProcessorList
import torch

SOT_TOKEN = "<|reserved_special_token_0|>"
EOT_TOKEN = "<|reserved_special_token_1|>"

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_sequence = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tokens):
        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end_of_sequence = True


    def get_next_tokens(self, tokens):
        node = self.root
        for token in tokens:
            if token not in node.children:
                return []
            node = node.children[token]
        return list(node.children.keys())

# Define a custom LogitsProcessor to constrain outputs to a target list
class CitationDecodingProcessor(LogitsProcessor):
    def __init__(self, tokenizer, target_sentences):
        self.tokenizer = tokenizer
        # Encode the target sentences into token IDs
        self.target_ids = [tokenizer.encode(target + EOT_TOKEN, add_special_tokens=False) for target in
                           target_sentences]
        # Convert to tensors for comparison
        self.target_ids_tensors = [torch.tensor(ids) for ids in self.target_ids]

        self.cite_start_token_id = tokenizer.convert_tokens_to_ids(SOT_TOKEN)
        self.cite_end_token_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
        self.trie = Trie()
        for target in self.target_ids:
            self.trie.insert(target)

    def _get_citation_start_pos(self, input_ids):
        for i in range(len(input_ids))[::-1]:
            if input_ids[i] == self.cite_start_token_id:
                return i
            elif input_ids[i] == self.cite_end_token_id:
                return -1
        return -1

    def __call__(self, input_ids, scores):
        """
        This method modifies the logits to constrain the outputs.
        """
        is_vllm = False
        if isinstance(input_ids, tuple):
            is_vllm, batch_size, seq_len = True, 1, len(input_ids)
            input_ids = torch.tensor(input_ids, device=scores.device).unsqueeze(0)
            scores = scores.unsqueeze(0)
        else:
            batch_size, seq_len = input_ids.shape


        # For each beam or batch, keep only target tokens
        for i in range(batch_size):
            citation_start_pos = self._get_citation_start_pos(input_ids[i])
            if citation_start_pos != -1:
                pos_within_citation = seq_len - citation_start_pos - 1
                current_tokens = input_ids[i][citation_start_pos + 1:].tolist()
                next_tokens = self.trie.get_next_tokens(current_tokens)
                valid_token_set = set(next_tokens)
                if len(valid_token_set) == 0:
                    valid_token_set.add(self.cite_end_token_id)

                if valid_token_set:
                    mask = torch.full(scores[i].shape, float('-inf'), device=input_ids.device)
                    mask[list(valid_token_set)] = 0.0
                    scores[i] += mask


        # if scores all -inf, or nan in scores, raise error
        if torch.all(scores == float('-inf')) or torch.any(torch.isnan(scores)):
            raise ValueError("All logits are -inf. Please check the model and the target sentences")

        if is_vllm:
            return scores[0]
        return scores


class RestrictTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, allowed_tokens):
        self.allowed_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in allowed_tokens]

    def __call__(self, input_ids, scores):
        # Set logits for all tokens not in the allowed list to a large negative value
        disallowed_token_mask = torch.ones_like(scores).bool()
        disallowed_token_mask[:, self.allowed_token_ids] = False
        scores[disallowed_token_mask] = -float('Inf')
        return scores