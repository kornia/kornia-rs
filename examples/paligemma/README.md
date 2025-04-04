Example showing how to generate a caption of an image using Google Paligemma.

```bash
Usage: paligemma -i <image-path> -p <text-prompt> [--sample-length <sample-length>]

Generate a description of an image using Google Paligemma

Options:
  -i, --image-path  path to an input image
  -p, --text-prompt prompt to ask the model
  --sample-length   the length of the generated text
  --help, help      display usage information
```

![xucli_sepas](https://github.com/user-attachments/assets/388ea039-d024-4a19-8462-f658856043b9)

```bash
cargo run --bin paligemma --release -- -i /home/edgar/Downloads/xucli_sepas.png -p "caption" --sample-length 100
```
or with cuda pass the flag `--features cuda`

```bash
caption. A cartoon of two men sitting under a red umbrella. One man wears sunglasses and has a gray beard, while the other man wears a black jacket and has a bald head. The man in the purple hoodie is smiling and holding a pen. The man in the black jacket has his hand on his leg and is looking at the camera. A green tree is in the distance.
77 tokens generated (18.58 token/s)
```
