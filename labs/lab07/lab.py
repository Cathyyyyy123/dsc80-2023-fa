# lab.py


import pandas as pd
import numpy as np
import os 
from pathlib import Path
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def match_1(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    pattern = r'^.{2}\[.{2}\]'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_2(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    pattern = r'^\(858\) \d{3}-\d{4}$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_3(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    pattern = r'^[A-Za-z0-9? ]{5,9}\?$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False
    >>> match_4("$iiuABc")
    False
    >>> match_4("123$$$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    >>> match_4("$!@$")
    False
    """
    pattern = r'^\$[^a-c]*\$(a|A)+(b|B)+(c|C)+'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    """
    pattern = r'^\w+\.py$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """
    pattern = r'^[a-z]+_{1}[a-z]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_7(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """
    pattern = r'^_[a-z]+_$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_8(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo!!!!!!! !!!!!!!!!")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """
    pattern = r'^[^iO1]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_9(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    >>> match_9('NY-32-LAX-0000') # If the state is NY, the city can be any 3 letter code, including LAX or SAN!
    True
    >>> match_9('TX-32-SAN-4491')
    False
    '''
    pattern = r'^(NY-\d{2}-[A-Z]{3}-\d{4}|(CA-\d{2}-(LAX|SAN)-\d{4}))'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_10('ABCdef')
    ['bcd']
    >>> match_10(' DEFaabc !g ')
    ['def', 'bcg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10('Ab..DEF')
    ['bde']
    
    '''
    string = string.lower()
    string = re.sub(r'[^\w]|a', '', string)
    result = []
    for i in range(0, len(string) - 2, 3):
        substring = string[i:i+3]
        if substring not in result:
            result.append(substring)
    return result


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def extract_personal(s):
    email = re.findall(r'\b\w+@\w+\.\w+\b', s)
    ssn = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', s)
    bitaddre = re.findall(r'\bbitcoin:\w+\b', s)
    streetadd = re.findall(r'\d{1,6} [A-Za-z0-9 ]+', s)
    return (email, ssn, bitaddre, streetadd)



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def tfidf_data(reviews_ser, review):
    tfidf_dict = {}
    words = review.strip().split()
    cnt = pd.Series(words).value_counts()
    for word in words:
        re_pat = fr'\b{word}\b'
        tf = len(re.findall(re_pat, review.strip())) / len(review.strip().split())
        idf = np.log(len(reviews_ser) / reviews_ser.str.contains(re_pat, regex=True).sum())
        tfidf_dict[word] = {'cnt': cnt[word], 'tf': tf, 'idf': idf, 'tfidf': tf * idf}
    tfidf = pd.DataFrame.from_dict(tfidf_dict, orient='index')
    return tfidf


def relevant_word(out):
    return out['tfidf'].idxmax()


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def extract_hashtags(tweet):
    return re.findall(r'#(\w+)', tweet)


def hashtag_list(tweet_text):
    return tweet_text.apply(extract_hashtags)


def most_common_hashtag(tweet_lists):
    all_hashtags = []
    result = []
    for sublist in tweet_lists.dropna():
        for hashtag in sublist:
            all_hashtags.append(hashtag)
    hashtag_counts = pd.Series(all_hashtags).value_counts()
    most_common_hashtag = hashtag_counts.idxmax()
    
    for out in tweet_lists:
        if len(out) == 0:
            result.append(np.nan)
        elif len(out) == 1:
            result.append(out[0])
        elif len(out) > 1:
            result.append(most_common_hashtag)
    return pd.Series(result)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def num_hashtags(content):
    return len(re.findall(r'#[A-Za-z0-9]+', content))


def clean_text(text):
    text = re.sub(r'@\w+|https?://\S+|#[A-Za-z0-9]+|RT', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def create_features(ira):
    ira['num_hashtags'] = ira['text'].apply(num_hashtags)
    content = hashtag_list(ira['text'])
    ira['mc_hashtags'] = most_common_hashtag(content)
    ira['num_tags'] = ira['text'].str.count(r'@[A-Za-z]+')
    ira['num_links'] = ira['text'].str.count(r'(http://\S+)|(https://\S+)')
    ira['is_retweet'] = ira['text'].str.startswith('RT').astype(int) != 0
    ira['text'] = ira['text'].apply(clean_text)
    return ira
