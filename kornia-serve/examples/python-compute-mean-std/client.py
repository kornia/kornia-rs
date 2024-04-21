"""Example of a client that sends a request to the server to compute the mean and std of a dataset."""

import argparse
import asyncio
from pathlib import Path
import httpx


async def main() -> None:
    """Main function to send a request to the server to compute the mean and std of a dataset."""
    parser = argparse.ArgumentParser(description="Compute mean and std of a dataset")
    parser.add_argument(
        "--images-dir", type=Path, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="Number of threads to use"
    )
    args = parser.parse_args()

    # create the request
    params = {
        "images_dir": str(args.images_dir),
        "num_threads": args.num_threads,
    }

    # send the request
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(
                "http://0.0.0.0:3000/api/v0/compute/mean-std", json=params
            )
        except httpx.HTTPError as _:
            print("The request timed out. Please try again.")
            return

    # parse the response
    json_response = response.json()
    print(json_response)


if __name__ == "__main__":
    asyncio.run(main())
