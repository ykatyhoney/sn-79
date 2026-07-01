# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
#!/usr/bin/env python

# Tool for generating artificial remote agent response structures for testing purposes.

import itertools as it
import json
from pathlib import Path
import random

import click

#--------------------------------------------------------------------------

@click.group()
def main():
    pass

#--------------------------------------------------------------------------

@main.command(no_args_is_help=True)
@click.option("--count", "-c", type=click.IntRange(min=1, max=1024), required=True,
              help="Number of (non-matching) orders to generate on both sides")
@click.option("--agents", "-a", type=click.IntRange(min=1), default=1, show_default=True,
              help="Number of agents for which to generate responses")
@click.option("--books", "-b", type=click.IntRange(min=1), default=1, show_default=True,
              help="Number of books for which to generate responses")
@click.option("--depth", "-d", type=click.IntRange(min=1), default=21, show_default=True,
              help="Maximum book depth")
@click.option("--shuffle", "-s", is_flag=True, default=True, show_default=True,
              help="Randomly shuffle the generated response array")
@click.option("--responses-file", "-rf", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Responses output file path")
@click.option("--state-file", "-sf", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="State output file path")
@click.option("-y", is_flag=True, default=False, show_default=True,
              help="Bypass all sanity prompts")
def create_responses(**opts):
    """
    Creates a file with a facsimile remote agent response structure.
    """
    COUNT, AGENTS, BOOKS = [opts[key] for key in ["count", "agents", "books"]]
    TOTAL_COUNT = COUNT * 2

    # Sanity checks.
    if not opts["y"] and TOTAL_COUNT * AGENTS * BOOKS > 10_000:
        if not click.confirm("Estimated agent response count over 10'000. Proceed?", default=None):
            click.echo("Exiting gracefully.")
            exit(0)

    # Generate the responses.
    responses_json = {}
    responses = []
    for direction, _, agent_id, book_id in it.product([0, 1], range(COUNT), range(AGENTS), range(BOOKS)):
        responses.append({
            "agentId": agent_id,
            "delay": 0,
            "type": "PLACE_ORDER_LIMIT",
            "payload": {
                "direction": direction,
                "volume": round(random.uniform(1.0, 2.0), 8),
                "price": round(random.uniform(0.5, 3.5) if direction == 0 else random.uniform(6.0, 10.0), 8),
                "bookId": book_id,
                "clientOrderId": None
            }
        })
    if opts["shuffle"]:
        random.shuffle(responses)
    responses_json["responses"] = responses

    # Generate the reference state.
    books = {book_id: {"bid": [], "ask": []} for book_id in range(BOOKS)}
    record = {book_id: [] for book_id in range(BOOKS)}
    for idx, res in enumerate(responses):
        direction, volume, price, book_id = \
            [res["payload"][key] for key in ["direction", "volume", "price", "bookId"]]
        books[book_id]["bid" if direction == 0 else "ask"].append({
            "price": price,
            "volume": volume,
            "orders": [{
                "orderId": idx,
                "timestamp": 0,
                "volume": volume,
                "direction": direction,
                "clientOrderId": None
            }]
        })
        record[book_id].append({
            "agentId": res["agentId"],
            "clientOrderId": None,
            "direction": direction,
            "event": "place",
            "orderId": idx,
            "price": price,
            "timestamp": 0,
            "volume": volume
        })
    state_json = {
        "books": [{
            "bookId": book_id,
            "record": record[book_id],
            "bid": sorted(books[book_id]["bid"], key=lambda x: x["price"], reverse=True),
            "ask": sorted(books[book_id]["ask"], key=lambda x: x["price"]),
        } for book_id in range(BOOKS)]
    }

    # Write results.
    opts["responses_file"].write_text(json.dumps(responses_json, indent=4, sort_keys=True))
    opts["state_file"].write_text(json.dumps(state_json, indent=4, sort_keys=True))

    click.echo("Done.")

#--------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#--------------------------------------------------------------------------
