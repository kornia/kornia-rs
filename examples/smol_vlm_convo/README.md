# SmolVLM Conversational / Chatbot-style Example

A crude example of using SmolVLM model in a traditional chatbot style.

![GIF](https://private-user-images.githubusercontent.com/39780709/485829240-b60b0328-13a7-4b46-b499-3d1aa1dcb1a9.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg3NzUzNTMsIm5iZiI6MTc1ODc3NTA1MywicGF0aCI6Ii8zOTc4MDcwOS80ODU4MjkyNDAtYjYwYjAzMjgtMTNhNy00YjQ2LWI0OTktM2QxYWExZGNiMWE5LmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTI1VDA0MzczM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTQ2OTFlYTdlYjAyZGViYzZmMTY0MDM3NjYzMWMwODMwMzk3NTA2NjM4ZDZkNzIxYjcxMjU5NTk2NjU3YWEwNDcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.oaz8RVapuiOdCcv69VlgqbVHrJThxhARyHSgfw2lOBw)

## Commands

```bash
cargo run -p smol_vlm_convo --features cuda
```
Options are configured via the UI.

## User Interface

The left panel can be accessed by entering into config mode via [Tab]. The center top panel is the chat history where the highlighted text is SmolVLM's response and the bottom panel is where you enter your prompt and, optionally, with an image. You can scroll through the history with top/down arrow.
