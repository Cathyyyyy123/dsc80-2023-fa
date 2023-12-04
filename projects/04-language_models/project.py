# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    response = requests.get(url)
    content = response.text.replace('\r\n', '\n')

    start_match = re.search(r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*', content)
    end_match = re.search(r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*', content)

    book_content = content[start_match.end():end_match.start()]
    return book_content


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    token_pattern = r'\w+|[^\w\s]'
    book_string = '\n\n' + book_string.strip() + '\n\n'
    def add_tokens(match):
        paragraph = match.group().strip()
        if paragraph:
            paragraph_tokens = re.findall(token_pattern, paragraph)
            return '\x02 ' + ' '.join(paragraph_tokens) + ' \x03'
        return ''

    tokenized_string = re.sub(r'\n\n+', add_tokens, book_string)
    tokens = re.findall(token_pattern, tokenized_string)
    # Ensure the list starts with START token and ends with STOP token
    if tokens[0] != '\x02':
        tokens.insert(0, '\x02')
    if tokens[-1] != '\x03':
        tokens.append('\x03')
    
    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
    
        unique_tokens = set(tokens)
        total_unique_tokens = len(unique_tokens)
        probability = 1 / total_unique_tokens
        token_probabilities = {token: probability for token in unique_tokens}
        return pd.Series(token_probabilities)
    
    def probability(self, words):

        prop = 1
        for word in words:
            p = self.mdl.get(word, 0)
            prop *= p
        return prop
        
    def sample(self, M):

        if self.mdl.empty:
          return ""
        random_tokens = np.random.choice(self.mdl.index, size=M, replace=True)
        return ' '.join(random_tokens)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        self.N = 1
        self.mdl = self.train(tokens)
    
    def train(self, tokens):

        token_counts = pd.Series(tokens).value_counts()
        token_probabilities = token_counts / token_counts.sum()
        return token_probabilities
    
    def probability(self, words):

        prob = 1.0
        for token in words:
            if token not in self.mdl:
                return 0
            prob *= self.mdl[token]
        return prob
        
    def sample(self, M):

        return ' '.join(np.random.choice(self.mdl.index, size=M, p=self.mdl.values, replace=True))


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        ngrams = []
        for i in range(len(tokens) - self.N + 1):
            ngram = tuple(tokens[i:i + self.N])
            ngrams.append(ngram)
        return ngrams

        
    def train(self, ngrams):
        # N-Gram counts C(w_1, ..., w_n)
        # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        # Create the conditional probabilities
        # Put it all together
        ngram_counts = pd.Series(ngrams).value_counts()
        
        n1gram_counts = pd.Series([ngram[:-1] for ngram in ngrams]).value_counts()
        
        probabilities = {ngram: count / n1gram_counts[ngram[:-1]] for ngram, count in ngram_counts.items()}
        prob_df = pd.DataFrame(list(probabilities.items()), columns=['ngram', 'prob'])
        prob_df['n1gram'] = prob_df['ngram'].apply(lambda x: x[:-1])
        prob_df = prob_df[['ngram', 'n1gram', 'prob']]
        return prob_df
        
    def probability(self, words):
        if isinstance(words, list):
            words = tuple(words)
        elif isinstance(words, str):
            words = (words,)

        prob = 1.0

        if len(words) == 1:
            if words[0] not in self.prev_mdl.mdl:
                return 0.0
            unigram_prob = self.prev_mdl.mdl[words[0]]
            return unigram_prob

        for i in range(len(words)):
            # ngram = words[i:i + self.N]
            ngram = words[max(0, i - self.N + 1):i + 1]
            # print(ngram)

            current_model = self
            while len(ngram) < current_model.N and current_model.prev_mdl is not None:
                current_model = current_model.prev_mdl
                # print(current_model)

            if isinstance(current_model.mdl, pd.Series):
                # Unigram case
                prob *= current_model.mdl.get(ngram[0], 0)  # Default to 0 if not found
                # print('unigram', prob)
            elif isinstance(current_model.mdl, pd.DataFrame):
                # Bigram or higher
                if ngram not in current_model.mdl['ngram'].tolist():
                    return 0.0
                prob *= current_model.mdl.loc[current_model.mdl['ngram'] == ngram, 'prob'].values[0]
                # print('bigram or higher', prob, current_model.mdl.loc[current_model.mdl['ngram'] == ngram, 'prob'].values[0])
            else:
                raise TypeError("Model is neither a Series nor a DataFrame")

        return prob
    
    
    
    def sample(self, M):
        # Use a helper function to generate sample tokens of length `length`
        def get_next_token(current_context, model_df=self.mdl):
            # possible_next_tokens = {}
            # context = tuple(current_context[-(self.N - 1):])
            # print(f'Context used for matching: {context}')

            # if context == ('\x02',) or '\x03' in context:
            #     context = ('\x02',)
            #     for _, row in model_df.iterrows():
            #         if row['n1gram'][0] == '\x02':
            #             possible_next_tokens[row['ngram'][1]] = row['prob']
            #     # print(f'1st Possible next tokens: {possible_next_tokens}')
            # else:
            #     for _, row in model_df.iterrows():
            #         if row['n1gram'] == context:
            #             possible_next_tokens[row['ngram'][-1]] = row['prob']
            #     # print(f'Possible next tokens: {possible_next_tokens}')
            context = tuple(current_context[-(self.N - 1):])

        # Filter the DataFrame based on the context for efficiency
            if context == ('\x02',) or '\x03' in context:
                filtered_df = model_df[model_df['n1gram'].apply(lambda x: x[0] == '\x02')]
            else:
                filtered_df = model_df[model_df['n1gram'] == context]

            # Create a dictionary of possible next tokens from the filtered DataFrame
            possible_next_tokens = {row['ngram'][-1]: row['prob'] for _, row in filtered_df.iterrows()}

            if not possible_next_tokens:
                return '\x03'

            tokens, probs = zip(*possible_next_tokens.items())
            # print(tokens, probs)
            tokens = np.array(tokens)
            normalized_probs = np.array(probs) / sum(probs)
            next_token = np.random.choice(tokens, p=normalized_probs)
            # print(f'Next token: {next_token}')

            return next_token
           
        
        # Transform the tokens to strings
        sequence = ['\x02']
        while len(sequence) < M + 1:
            next_token = get_next_token(sequence)
            sequence.append(next_token)
            if next_token == '\x03':
                break
        
        while len(sequence[1:]) < M:
            sequence.append('\x03')

        return ' '.join(sequence)