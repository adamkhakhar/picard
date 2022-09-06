from typing import List


def tokenize(sql_tokens: List[str], probs: List[float], prob_combine_fun=lambda l: sum(l) / len(l)) -> List[str]:
    # print("sql_tokens", sql_tokens)
    # remove all tokens which are before select token
    for i in range(len(sql_tokens)):
        curr_tok = sql_tokens[i]
        if curr_tok.lower() == "select":
            sql_tokens = sql_tokens[i:]
            probs = probs[i:]
            break
    clean_toks = []
    clean_probs = []
    # remove all tokens with no characters or whitespace
    for i in range(len(sql_tokens)):
        # rm whitespace, empty
        if sql_tokens[i].isspace() or len(sql_tokens[i]) == 0:
            continue
        clean_toks.append(sql_tokens[i].lower())
        clean_probs.append(probs[i])

    assert len(clean_toks) == len(clean_probs)
    # print("clean_toks", clean_toks)

    combined_toks = []
    combined_probs = []
    curr_tok = ""
    curr_prob = []
    in_partial_tok = False
    # combine tokens that are not sql syntax
    sql_syntax = set(
        [
            "(",
            ")",
            "*",
            "select",
            "having",
            "where",
            "group",
            "by",
            "limit",
            "order",
            ",",
            "+",
            "-",
            "/",
            "join",
            "inner",
            "outer",
            "on",
            "distinct",
            ";",
            "max",
            "min",
            "count",
            "sum",
            "avg",
            "not",
            "between",
            "=",
            ">",
            "<",
            ">=",
            "<=",
            "!=",
            "in",
            "like",
            "is",
            "exists",
            "join",
            "on",
            "as",
            "select",
            "from",
            "where",
            "group",
            "order",
            "limit",
            "intersect",
            "union",
            "except",
            "and",
            "or",
            "desc",
            "asc",
        ]
    )
    for i in range(len(clean_toks)):
        if clean_toks[i] in sql_syntax:
            # add current partial to combined
            if in_partial_tok:
                combined_toks.append(curr_tok)
                combined_probs.append(prob_combine_fun(curr_prob))
                curr_tok = ""
                curr_prob = []
            combined_probs.append(clean_probs[i])
            combined_toks.append(clean_toks[i])
            in_partial_tok = False
        else:
            curr_tok += clean_toks[i]
            curr_prob.append(clean_probs[i])
            in_partial_tok = True

        # handle case where last is not in syntax
        if i == len(clean_toks) - 1 and in_partial_tok:
            combined_toks.append(curr_tok)
            combined_probs.append(prob_combine_fun(curr_prob))

    # print("combined_toks", combined_toks)
    # print("combined_probs", combined_probs)

    return combined_toks
