Example showing how to use the Google Paligemma model with Candle.
## Setup

Setup huggingface account credentials to access the model.

 Accept the licence once (in a browser)

    Open https://huggingface.co/google/paligemma-3b-mix-224

    Log in and click “Agree and access”.

    You should now see a green tick “You have access”.

Create / copy a user access-token

    Go to Settings ▸ Access Tokens on Hugging Face.

    Click “New token”, scope: Read is enough. Copy the string that looks like hf_abcd….

Tell the CLI about the token:

    huggingface-cli login --token hf_yourtoken
## Usage

```bash
Usage: paligemma -i <image-path> -p <text-prompt> [--sample-length <sample-length>]

Generate a description of an image using Google Paligemma

Options:
  -i, --image-path  path to an input image
  -p, --text-prompt prompt to ask the model
  --sample-length   the length of the generated text
  --help, help      display usage information
```

```bash
cargo run -p paligemma -- -i ./data/gangsters.png -p "cap en" --sample-length 100
```
or with cuda pass the flag `--features cuda`


![xucli_sepas](https://github.com/user-attachments/assets/388ea039-d024-4a19-8462-f658856043b9)

```bash
cap enTwo men are sitting under an umbrella, the left man is wearing sunglasses.
16 tokens generated (26.15 token/s)
```
