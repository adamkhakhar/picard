from pydantic import BaseModel
from sqlite3 import Connection, connect, OperationalError
from fastapi import HTTPException
import os


class AskResponse(BaseModel):
    query: str
    execution_results: list


def response(query: str, conn: Connection) -> AskResponse:
    try:
        return AskResponse(query=query, execution_results=conn.execute(query).fetchall())
    except OperationalError as e:
        raise HTTPException(
            status_code=500, detail=f'while executing "{query}", the following error occurred: {e.args[0]}'
        )


class SpiderDB:
    """
    Query Spider Database
    """

    def __init__(self, path_to_database=os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/database"):
        self.path_to_database = path_to_database

    def query(self, db_id, query):
        conn = connect(self.path_to_database + "/" + db_id + "/" + db_id + ".sqlite")
        return response(query=query, conn=conn).execution_results
