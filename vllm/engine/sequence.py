

class Sequence:
    def __init__(self, prompt, token_ids):
        self.block_list = []
        self.prompt = prompt
        self.token_ids = token_ids
        self.num_hashed_tokens = 0
        self.num_scheduled_tokens = len(token_ids)
        self.num_computed_tokens = 0

    def __len__(self):
        return len(self.token_ids)


    def reset(self):
        self.block_list.clear()
        self.num_hashed_tokens = 0
        self.num_computed_tokens = 0