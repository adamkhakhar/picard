from typing import List


def tokenize(sql_tokens: List[str]) -> List[str]:
    print("sql_tokens", sql_tokens)
    clean_toks = []
    # remove all tokens which are before select token
    select_included = False
    for i in range(len(sql_tokens)):
        curr_tok = sql_tokens[i]
        if curr_tok.lower() == "select":
            select_included = True
            sql_tokens = sql_tokens[i:]
            break
    # remove white space tokens
    for curr_tok in sql_tokens:
        if not curr_tok.isspace() and len(curr_tok) > 0:
            clean_toks.append(curr_tok)
    print("clean_toks", clean_toks)
    return clean_toks
