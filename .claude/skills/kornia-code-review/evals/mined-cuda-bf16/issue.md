title:	kornia-vlm: FP16 fallback warning should say why FP16 can be worse
state:	OPEN
author:	contributor
labels:	enhancement
number:	9995
--
When a GPU lacks BF16, `get_device_and_dtype` falls back to FP16 with a warning that does not say what the user loses. Reword it: FP16 may overflow in attention logits and give lower-quality results.
