from adt import *
from typing import *
import dataclasses
import itertools


def record(
    *args,
    init=True,
    repr=True,
    eq=True,
    order=True,
    unsafe_hash=False,
    **kwargs,
):
    return dataclasses.dataclass(
        *args, init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=True, **kwargs
    )


AggOp = Any


@record
class PCol:
    table_name: str
    col_name: str

    def __str__(self):
        return self.table_name + "." + self.col_name


@record
class PAggCol:
    col: PCol
    agg_op: AggOp

    def __str__(self):
        return f" {self.agg_op}({self.col})"


@adt
class PCondExpr:
    COLUMN: Case[PCol]
    LITERAL_STRING: Case[str]
    LITERAL_INT: Case[int]
    LITERAL_FLOAT: Case[float]
    SQL: Case["PSQL"]

    def __str__(self):
        return self.match(
            column=lambda col: f"{col}",
            literal_string=lambda x: f'"{x}"',
            literal_int=lambda x: f"{x}",
            literal_float=lambda x: f"{x}",
            sql=lambda _: f"[NESTED_SQL]",
        )


@adt
class PWhereCond:
    NOT_: Case[PCol]
    BETWEEN: Case[PCol, PCondExpr, PCondExpr]
    EQ: Case[PCol, PCondExpr]
    GT: Case[PCol, PCondExpr]
    LT: Case[PCol, PCondExpr]
    GTEQ: Case[PCol, PCondExpr]
    LTEQ: Case[PCol, PCondExpr]
    NEQ: Case[PCol, PCondExpr]
    IN_: Case[PCol, PCondExpr]
    LIKE: Case[PCol, PCondExpr]
    EXISTS: Case[PCol, PCondExpr]

    def __str__(self):
        return self.match(
            not_=lambda col: f"!{col}",
            between=lambda col, col2, col3: f"{col} BETWEEN {col2} AND {col3}",
            eq=lambda col, col2: f"{col} = {col2}",
            gt=lambda col, col2: f"{col} > {col2}",
            lt=lambda col, col2: f"{col} < {col2}",
            gteq=lambda col, col2: f"{col} >= {col2}",
            lteq=lambda col, col2: f"{col} <= {col2}",
            neq=lambda col, col2: f"{col} != {col2}",
            in_=lambda col, col2: f"[IN]",
            like=lambda col, col2: f"[LIKE]",
            exists=lambda col, col2: f"[EXISTS]",
        )


@adt
class PHavingCond:
    NOT_: Case[PAggCol]
    BETWEEN: Case[PAggCol, PCondExpr, PCondExpr]
    EQ: Case[PAggCol, PCondExpr]
    GT: Case[PAggCol, PCondExpr]
    LT: Case[PAggCol, PCondExpr]
    GTEQ: Case[PAggCol, PCondExpr]
    LTEQ: Case[PAggCol, PCondExpr]
    NEQ: Case[PAggCol, PCondExpr]
    IN_: Case[PAggCol, PCondExpr]
    LIKE: Case[PAggCol, PCondExpr]
    EXISTS: Case[PAggCol, PCondExpr]

    def __str__(self):
        return self.match(
            not_=lambda col: f"!{col}",
            between=lambda col, col2, col3: f"{col} BETWEEN {col2} AND {col3}",
            eq=lambda col, col2: f"{col} = {col2}",
            gt=lambda col, col2: f"{col} > {col2}",
            lt=lambda col, col2: f"{col} < {col2}",
            gteq=lambda col, col2: f"{col} >= {col2}",
            lteq=lambda col, col2: f"{col} <= {col2}",
            neq=lambda col, col2: f"{col} != {col2}",
            in_=lambda col, col2: f"[IN]",
            like=lambda col, col2: f"[LIKE]",
            exists=lambda col, col2: f"[EXISTS]",
        )


@record
class PSQL:
    select_distinct: bool
    select_cols: List[PAggCol]  # list of agg ops and col op things
    from_tables: List["PTableExpr"]
    from_conds: List[Tuple[PCol, PCol]]  # list of pairs of columns for joining
    where_conds: List[PWhereCond]
    having_conds: List[PHavingCond]
    groupby_cols: List[PCol]
    orderby_desc: bool
    orderby_cols: List[PAggCol]
    limit: int
    nested_except: Optional["PSQL"]
    nested_union: Optional["PSQL"]
    nested_intersect: Optional["PSQL"]

    def __str__(self):
        r = "SELECT "
        if self.select_distinct:
            r += "DISTINCT "

        r += "[" + "; ".join(str(x) for x in self.select_cols) + "]"

        r += "FROM "
        r += "["
        for table_expr in self.from_tables:
            r += table_expr.match(
                table=lambda name: f"{name};",
                sql=lambda psql2: f"[NESTED SQL]",
            )
        r += "] "

        if len(self.from_conds) > 0:
            r += "[FROM_CONDS]"

        if len(self.where_conds) > 0:
            r += "WHERE " + " ".join(str(x) for x in self.where_conds)

        if len(self.groupby_cols) > 0:
            r += "GROUPBY "
            r += "[" + "; ".join(str(x) for x in self.groupby_cols) + "]"

        if len(self.having_conds) > 0:
            r += "HAVING " + " ".join(str(x) for x in self.having_conds)

        if len(self.orderby_cols) > 0:
            r += "ORDERBY "
            r += "DESC " if self.orderby_desc else "ASC "
            r += "[" + "; ".join(str(x) for x in self.groupby_cols) + "]"

        if self.limit is not None:
            r += f"LIMIT {self.limit}"

        if self.nested_except is not None:
            r += "[NESTED EXCEPT]"

        if self.nested_union is not None:
            r += "[NESTED UNION]"

        if self.nested_intersect is not None:
            r += "[NESTED INTERSECT]"
        return r


@adt
class PTableExpr:
    TABLE: Case[str]
    SQL: Case[PSQL]

    def __str__(self):
        return self.match(
            table=lambda table: f"{table}",
            sql=lambda _: f"[NESTED_SQL]",
        )


WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")


def parse_col(db_tables, col_idx):
    table_idx, col_name = None, None
    if type(col_idx) == int:
        table_idx, col_name = db_tables["column_names_original"][col_idx]
    else:
        col_lower = "*" if col_idx is "__all__" else col_idx.split(".")[1][:-2].lower()
        table_idx, col_name = None, None
        for table_info in db_tables["column_names_original"]:
            if table_info[1].lower() == col_lower:
                table_idx = table_info[0]
                col_name = table_info[1]
    if table_idx == -1:
        table_name = "_"
    else:
        table_name = db_tables["table_names_original"][table_idx]

    return PCol(table_name, col_name)


def parse_filterexpr(db_tables, clause):
    if type(clause) == list:
        clause_a, clause_b, clause_c = clause
        assert clause_a == 0, clause
        assert clause_c == False  # is this is_distinct?
        return PCondExpr.COLUMN(parse_col(db_tables, clause_b))
    elif type(clause) == dict:
        return PCondExpr.SQL(parse_sql(db_tables, clause))
    elif type(clause) == str:
        assert clause[0] == '"'
        assert clause[-1] == '"'
        return PCondExpr.LITERAL_STRING(clause[1:-1])
    elif type(clause) == int:
        return PCondExpr.LITERAL_INT(clause)
    elif type(clause) == float:
        return PCondExpr.LITERAL_FLOAT(clause)
    else:
        assert False, clause


def parse_from_conds(db_tables, from_conds):
    p_from_conds = []
    and_count = 0
    for clause in from_conds:
        if type(clause) == list:
            negated, op_id, lhs, rhs1, rhs2 = clause
            (
                lhs_unit_op_id,
                (
                    lhs_agg_op_id,
                    lhs_col_id,
                    lhs_distinct,
                ),
                lhs_col_unit2,
            ) = lhs
            assert negated == False
            assert lhs_unit_op_id == 0
            assert lhs_col_unit2 is None
            assert lhs_distinct == False
            assert lhs_agg_op_id == 0
            assert op_id == 2  # 2 is "=". i think this is for joining on columns

            rhs1_a, rhs1_b, rhs1_c = rhs1
            assert rhs1_a == 0, rhs1
            assert rhs1_c == False  # is this is_distinct?

            assert rhs2 is None
            p_from_conds.append(
                (
                    parse_col(db_tables, lhs_col_id),
                    parse_col(db_tables, rhs1_b),
                )
            )
        elif clause == "and":
            and_count += 1
        else:
            assert False, clause
    assert len(p_from_conds) == 0 or (and_count + 1 == len(p_from_conds))
    return list_as_set(p_from_conds)


class BinopUnsupportedException(Exception):
    pass


def parse_aggcol(db_tables, clause, extra_agg_op):
    (
        agg_op_idx,
        col_idx,
        distinct,
    ) = clause

    col = parse_col(db_tables, col_idx)
    agg_op = AGG_OPS[agg_op_idx]

    if extra_agg_op != "none":
        assert agg_op == "none"
        agg_op = extra_agg_op
    if distinct:
        assert agg_op == "count"
        agg_op = "count_distinct"

    if col == PCol("_", "*"):
        assert agg_op == "count" or agg_op == "none"

    return PAggCol(
        col,
        agg_op,
    )


def parse_unit_thing(db_tables, clause, extra_agg_op):
    (
        unit_op_idx,
        col_unit1,
        col_unit2,
    ) = clause
    if col_unit2 is not None:
        raise BinopUnsupportedException()
    else:
        assert unit_op_idx == 0
        return parse_aggcol(db_tables, col_unit1, extra_agg_op)


def seq_split(tok, l):
    if len(l) == 0:
        return tuple()

    ret = []
    cur = []

    for x in l:
        if x == tok:
            ret.append(tuple(cur))
            cur = []
        else:
            cur.append(x)

    ret.append(tuple(cur))

    return ret


def parse_wherehaving_conds(db_tables, clauses, is_having):
    rets = []
    for disjunct in seq_split("or", clauses):
        disjunct_ret = []
        for conjunct in seq_split("and", disjunct):
            (clause,) = conjunct
            negated, op_id, lhs, rhs1, rhs2 = clause

            plhs = parse_unit_thing(db_tables, lhs, "none")
            if not is_having:
                assert plhs.agg_op == "none"
                plhs = plhs.col

            def binop(ctor):
                assert rhs2 is None
                return ctor(
                    plhs,
                    parse_filterexpr(db_tables, rhs1),
                )

            cond_type = PHavingCond if is_having else PWhereCond
            op = WHERE_OPS[op_id]
            if op == "not":
                assert rhs1 is None
                assert rhs2 is None
                assert False, op
            elif op == "between":
                ret = cond_type.BETWEEN(
                    plhs,
                    parse_filterexpr(db_tables, rhs1),
                    parse_filterexpr(db_tables, rhs2),
                )
            elif op == "=":
                ret = binop(cond_type.EQ)
            elif op == ">":
                ret = binop(cond_type.GT)
            elif op == "<":
                ret = binop(cond_type.LT)
            elif op == ">=":
                ret = binop(cond_type.GTEQ)
            elif op == "<=":
                ret = binop(cond_type.LTEQ)
            elif op == "!=":
                ret = binop(cond_type.NEQ)
            elif op == "in":
                ret = binop(cond_type.IN_)
            elif op == "like":
                ret = binop(cond_type.LIKE)
            elif op == "exists":
                assert False, op
            else:
                assert False, op

            disjunct_ret.append(ret)
        rets.append(list_as_set(disjunct_ret))

    return list_as_set(rets)


def list_as_set(l):
    r = frozenset(l)
    assert len(r) == len(l), "; ".join(str(x) for x in l)
    return r


def parse_sql(db_tables, sql):
    p_from_tables = []
    for tu, tid in sql["from"]["table_units"]:
        if tu == "table_unit":
            if type(tid) == int:
                p_from_tables.append(PTableExpr.TABLE(db_tables["table_names_original"][tid]))
            else:
                p_from_tables.append(PTableExpr.TABLE(tid[2:-2]))
        elif tu == "sql":
            p_from_tables.append(PTableExpr.SQL(parse_sql(db_tables, tid)))
        else:
            assert False, tu

    from_conds = parse_from_conds(db_tables, sql["from"]["conds"])

    assert len(sql["select"]) == 2
    select_distinct = sql["select"][0]
    select_cols = []
    for clause in sql["select"][1]:
        (
            agg_op_idx,
            unit_clause,
        ) = clause
        agg_op = AGG_OPS[agg_op_idx]
        cad = parse_unit_thing(db_tables, unit_clause, agg_op)

        select_cols.append(cad)

    where_conds = parse_wherehaving_conds(db_tables, sql["where"], False)
    having_conds = parse_wherehaving_conds(db_tables, sql["having"], True)

    p_groupby = []
    for clause in sql["groupBy"]:
        aggcol = parse_aggcol(db_tables, clause, "none")
        assert aggcol.agg_op == "none"
        p_groupby.append(aggcol.col)

    if len(p_groupby) == 0:
        assert len(having_conds) == 0

    if len(sql["orderBy"]) == 0:
        orderby_desc = None
        p_orderby = []
    elif len(sql["orderBy"]) == 2:
        if sql["orderBy"][0] == "desc":
            orderby_desc = True
        elif sql["orderBy"][0] == "asc":
            orderby_desc = False
        else:
            assert False, sql["orderBy"][0]

        p_orderby = []
        for clause in sql["orderBy"][1]:
            p_orderby.append(parse_unit_thing(db_tables, clause, "none"))
    else:
        assert False

    return PSQL(
        select_distinct=select_distinct,
        select_cols=tuple(select_cols),
        from_tables=tuple(p_from_tables),
        from_conds=from_conds,
        where_conds=where_conds,
        having_conds=having_conds,
        groupby_cols=list_as_set(p_groupby),
        orderby_cols=tuple(p_orderby),
        orderby_desc=orderby_desc,
        limit=sql["limit"],
        nested_union=parse_sql(db_tables, sql["union"]) if sql["union"] is not None else None,
        nested_except=parse_sql(db_tables, sql["except"]) if sql["except"] is not None else None,
        nested_intersect=parse_sql(db_tables, sql["intersect"]) if sql["intersect"] is not None else None,
    )


def spidsl_parse(query):
    db_tables = spider_dataset.tables[query["db_id"]]
    sql = query["sql"]

    return parse_sql(db_tables, sql)


# blacklisted_qids = {
#     # TODO : why do these fail? I think it's an issue with me parsing table names in the context of joins
#     ("train", 1646),
#     ("train", 4482),
#     ("train", 4483),
# }
# blacklisted_dbs = {
#     # "car_1", # this database has some columns with the wrong types, which messes up our numpy evaluator since it doesn't auto-coerce types
# }
#
# dataset_all_whitelisted = {
#     qid : query
#     for qid, query in spider_dataset.data_all.items()
#     if (
#         qid not in blacklisted_qids
#     and query['db_id'] not in blacklisted_dbs
#     )
# }
#
#
# skipped_count = 0
# for qid, query in dataset_all_whitelisted.items():
#     try:
#         spidsl_parse(query)
#     except BinopUnsupportedException:
#         # print(f"skipping {qid} due to binop")
#         skipped_count += 1
#     except Exception as e:
#         print(qid)
#         print(query["query"])
#         print(query["sql"])
#         raise e
# print("parsing psql skipped", skipped_count/len(dataset_all_whitelisted))


##################3 BEGIN ssdsl.py ###################

from dataclasses import dataclass
from typing import *
from adt import *

FilterOp = Any


@adt
class Literal:
    LITERAL_STRING: Case[str]
    LITERAL_INT: Case[int]
    LITERAL_FLOAT: Case[float]

    def __getstate__(self):
        return (self._key.name, self._value)

    def __setstate__(self, p):
        self._key = Literal._Key[p[0]]
        self._value = p[1]


@dataclass(frozen=True)
class SSDSLQuery:
    table_names: Tuple[str]
    join_col_pairs: Tuple[Tuple[PCol, PCol]]
    where_conds: Tuple[Tuple[FilterOp, PCol, Literal]]
    grouping: Optional["SSDSLQueryGrouping"]


@dataclass(frozen=True)
class SSDSLQueryGrouping:
    group_col: Optional[PCol]  # None groups all
    aggregations: Set[PAggCol]
    having_conds: Tuple[Tuple[FilterOp, PAggCol, Literal]]


class SSDSLUnsupportedException(Exception):
    def __init__(self, args):
        self.info = args

    def __str__(self):
        return "\n".join(str(x) for x in self.info)


def raise_unsupported(*args):
    raise SSDSLUnsupportedException(args)


def assert_unsupported(b, *args):
    if not b:
        raise SSDSLUnsupportedException(args)


def collect_aggcols(expr: PSQL):
    ret = set()

    for select_aggcol in expr.select_cols:
        ret.add(select_aggcol)

    for disjunct in expr.having_conds:
        for having_cond in disjunct:
            ret.add(
                having_cond.match(
                    not_=lambda aggcol: aggcol,
                    between=lambda aggcol, col2, col3: aggcol,
                    eq=lambda aggcol, col2: aggcol,
                    gt=lambda aggcol, col2: aggcol,
                    lt=lambda aggcol, col2: aggcol,
                    gteq=lambda aggcol, col2: aggcol,
                    lteq=lambda aggcol, col2: aggcol,
                    neq=lambda aggcol, col2: aggcol,
                    in_=lambda aggcol, col2: aggcol,
                    like=lambda aggcol, col2: aggcol,
                    exists=lambda aggcol, col2: aggcol,
                )
            )
    return frozenset(ret)


def compile_spidsl_literal(pcondexpr: PCondExpr):
    return pcondexpr.match(
        column=raise_unsupported,
        literal_string=lambda x: Literal.LITERAL_STRING(x),
        literal_int=lambda x: Literal.LITERAL_INT(x),
        literal_float=lambda x: Literal.LITERAL_FLOAT(x),
        sql=raise_unsupported,
    )


# TODO: deleteme
class Counter(Exception):
    pass


def rcount(c):
    def _(*args):
        raise Counter()
        # assert c

    return _


# end deleteme


def compile_cond(cond):
    def handle_cond(op_name):
        return lambda lhs_col, rhs_expr: (
            op_name,
            lhs_col,
            compile_spidsl_literal(rhs_expr),
        )

    return cond.match(
        NOT_=raise_unsupported,
        BETWEEN=raise_unsupported,
        EQ=handle_cond("=="),
        GT=handle_cond(">"),
        LT=handle_cond("<"),
        GTEQ=raise_unsupported,
        LTEQ=raise_unsupported,
        NEQ=raise_unsupported,
        IN_=raise_unsupported,
        LIKE=raise_unsupported,
        EXISTS=raise_unsupported,
    )


# this deliberately ignores select, limit, and order
def compile_spidsl(expr: PSQL):
    for table in expr.from_tables:
        assert_unsupported(table._key == PTableExpr._Key.TABLE, "nested sql")

    assert_unsupported(len(expr.from_tables) == len(set(expr.from_tables)), "duplicate tables")
    assert_unsupported(expr.nested_except is None, "nesting")
    assert_unsupported(expr.nested_union is None, "nesting")
    assert_unsupported(expr.nested_intersect is None, "nesting")
    assert_unsupported(len(expr.where_conds) <= 1, "disjunctions")
    assert_unsupported(len(expr.having_conds) <= 1, "disjunctions")
    assert_unsupported(len(expr.groupby_cols) <= 1, "multigroup")

    aggcols = collect_aggcols(expr)
    nontrivial_aggcols = frozenset(aggcol for aggcol in aggcols if aggcol.agg_op != "none")
    is_grouping = len(nontrivial_aggcols) > 0 or len(expr.groupby_cols) > 0
    if not is_grouping:
        assert len(expr.groupby_cols) == 0

    ret_table_names = tuple(table_expr.table() for table_expr in expr.from_tables)

    # flattening disjunctions and conjunctions is only okay because we have at most one disjunct
    ret_where_conds = tuple(compile_cond(conjunct) for disjunct in expr.where_conds for conjunct in disjunct)

    return SSDSLQuery(
        table_names=ret_table_names,
        join_col_pairs=tuple(expr.from_conds),
        where_conds=ret_where_conds,
        grouping=compile_spidsl_grouping(expr) if is_grouping else None,
    )


def compile_spidsl_grouping(expr: PSQL):
    ret_group_col = None if len(expr.groupby_cols) == 0 else next(iter(expr.groupby_cols))

    ret_aggregations = frozenset(collect_aggcols(expr))

    # flattening disjunctions and conjunctions is only okay because we have at most one disjunct
    ret_having_conds = tuple(compile_cond(conjunct) for disjunct in expr.having_conds for conjunct in disjunct)

    return SSDSLQueryGrouping(
        group_col=ret_group_col,
        aggregations=ret_aggregations,
        having_conds=ret_having_conds,
    )


# n_succ = 0
# n_unsupp = 0
# n_counter = 0
# compiled_spider_dataset = {}
# for qid, query in dataset_all_whitelisted.items():
#     try:
#         psql = spidsl_parse(query)
#         ssexpr = compile_spidsl(psql)
#         n_succ += 1
#         compiled_spider_dataset[qid] = ssexpr
#     except SSDSLUnsupportedException:
#         n_unsupp += 1
#         pass
#     except BinopUnsupportedException:
#         n_unsupp += 1
#         pass
#     except Counter:
#         n_counter += 1
#
# print("SpiDSL to SSDSL compiler:", n_succ, n_unsupp, n_counter)


def ssdsl_to_sexpr(expr):
    ret = ("done",)

    if expr.grouping is not None:
        for op, aggcol, literal in reversed(expr.grouping.having_conds):
            ret = (
                "having",
                (op,),
                (aggcol,),
                (literal,),
                ret,
            )

        if expr.grouping.group_col is not None:
            ret = (
                "group_col",
                (expr.grouping.group_col,),
                (expr.grouping.aggregations,),
                ret,
            )
        else:
            ret = (
                "group_all",
                (expr.grouping.aggregations,),
                ret,
            )

    for op, col, literal in reversed(expr.where_conds):
        ret = (
            "where",
            (op,),
            (col,),
            (literal,),
            ret,
        )

    ret = (
        "done",
        ret,
    )

    for col1, col2 in reversed(expr.join_col_pairs):
        ret = (
            "join",
            (col1,),
            (col2,),
            ret,
        )

    return ret


base_dir = "/home/akhakhar/data/spider"
import json

with open(f"{base_dir}/tables.json", "r") as f:
    tables_json = json.load(f)
tables = {}
for table in tables_json:
    db_id = table["db_id"]
    assert db_id not in tables
    tables[db_id] = table


def spider_json_to_sexpr(spider_json, db_id):
    # db_tables = spider_dataset.tables[query["db_id"]]
    # sql = query["sql"]

    db_tables = tables[db_id]
    sql = spider_json

    spidsl_expr = parse_sql(db_tables, sql)
    ssdsl_expr = compile_spidsl(spidsl_expr)
    sexpr = ssdsl_to_sexpr(ssdsl_expr)
    return sexpr


if __name__ == "__main__":
    spider_json = {
        "from": {"table_units": [["table_unit", 1]], "conds": []},
        "select": [False, [[3, [0, [0, 0, False], None]]]],
        "where": [],
        "groupBy": [],
        "having": [],
        "orderBy": [],
        "limit": None,
        "intersect": None,
        "union": None,
        "except": None,
    }
    # print(spider_json_to_sexpr(spider_json, "concert_singer"))
    spider_json = {
        "from": {"table_units": [("table_unit", "__stadium__")], "conds": []},
        "select": (
            False,
            [(1, (0, (0, "__stadium.capacity__", False), None)), (0, (0, (0, "__stadium.average__", False), None))],
        ),
        "where": [],
        "groupBy": [],
        "having": [],
        "orderBy": [],
        "limit": None,
        "intersect": None,
        "union": None,
        "except": None,
    }
    # print(spider_json_to_sexpr(spider_json, "concert_singer"))
    # print("tables")
    # print([key for key in tables])
    # print(type(tables["college_2"]))
