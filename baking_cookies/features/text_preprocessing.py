import logging
import re
import nltk
import string
from nltk.corpus import stopwords
from itertools import chain
import baking_cookies

logger = logging.getLogger(__name__)

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english') +
                 list(string.punctuation) +
                 ['\\n'] + ['quot'])

regex_str = ["http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|"
             r"[!*\(\),](?:%[0-9a-f][0-9a-f]))+",
             r"(?:\w+-\w+){2}",
             r"(?:\w+-\w+)",
             r"(?:\\\+n+)",
             r"(?:@[\w_]+)",
             "<[^>]+>",
             r"(?:\w+'\w)",
             r"(?:[\w_]+)",
             r"(?:\S)"
             ]

# Create the tokenizer which will be case insensitive and will ignore space.
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')',
                       re.VERBOSE | re.IGNORECASE)


def tokenize_document(text, min_length=3, flatten=False):
    """Preprocess a whole raw document.

    Args:
        text (str): Raw string of text.
        min_length (int, optional): Minimum token length
        flatten (bool): Whether to flatten out sentences

    Returns:
        List: preprocessed and tokenized documents

    #UTILS
    """
    text = [clean_and_tokenize(sentence, min_length)
            for sentence in nltk.sent_tokenize(text)]
    if flatten:
        return list(chain(*text))
    else:
        return text


def clean_and_tokenize(text, min_length):
    """Preprocess a raw string/sentence of text.

    Args:
        text (str): Raw string of text.
        min_length (int): Minimum token length

    Returns:
        list of str: Preprocessed tokens.

    #UTILS
    """

    # Find tokens and lowercase
    tokens = tokens_re.findall(text)
    _tokens = [t.lower() for t in tokens]
    # Remove short tokens, stop words, tokens with digits, non-ascii chars
    filtered_tokens = [token.replace('-', '_') for token in _tokens
                       if len(token) >= min_length
                       and token not in stop_words
                       and not any(x in token for x in string.digits)
                       and any(x in token for x in string.ascii_lowercase)]
    return filtered_tokens

