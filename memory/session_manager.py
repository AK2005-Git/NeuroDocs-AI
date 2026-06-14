import json
import os


def save_session(
    session_name,
    history
):

    path = (
        f"data/chat_sessions/"
        f"{session_name}.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )