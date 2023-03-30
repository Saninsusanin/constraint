from functools import partial
from transformers.generation import Constraint


class ForbiddenNgramsConstraint(Constraint):
    def __init__(self, phrase: str, tokenizer):
        super(Constraint, self).__init__()
        self.phrase = phrase
        self.tokenizer = tokenizer
        self.possible_tokens = set(tokenizer.get_vocab().values())
        tokenizer = partial(tokenizer, add_special_tokens=False)
        self.tokens_per_word = [tokenizer(word).input_ids for word in phrase.split()]

        self.seqlen = 7
        self.completed = False

        self.curr_token_pos = 0
        self.number_of_completed_words = 0

        self.last_word_tokens = set(self.tokens_per_word[-1])

    def advance(self):
        if self.number_of_completed_words == len(self.tokens_per_word) - 1:
            return list(self.possible_tokens - self.last_word_tokens)
        else:
            return list(self.possible_tokens)

    def does_advance(self, token_id: int):
        if self.number_of_completed_words == len(self.tokens_per_word) - 1:
            return token_id not in self.last_word_tokens
        else:
            return True

    def update(self, token_id: int):
        curr_word_of_interest = self.tokens_per_word[self.number_of_completed_words]
        curr_token_id_of_iterest = curr_word_of_interest[self.curr_token_pos]

        if self.does_advance(token_id):
            if token_id == curr_token_id_of_iterest:
                self.curr_token_pos += 1

                if self.curr_token_pos == len(curr_word_of_interest):
                    self.curr_token_pos = 0
                    self.number_of_completed_words += 1

        stepped = True
        completed = False
        reset = False

        return stepped, completed, reset

    def reset(self):
        pass

    def remaining(self):
        return self.seqlen

    def copy(self, stateful=False):
        new_constraint = ForbiddenNgramsConstraint(self.phrase, self.tokenizer)

        if stateful:
            new_constraint.seqlen = self.seqlen
            new_constraint.curr_token_pos = self.curr_token_pos
            new_constraint.number_of_completed_words = self.number_of_completed_words

        return new_constraint
