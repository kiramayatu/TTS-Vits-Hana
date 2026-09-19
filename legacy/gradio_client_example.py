from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
result = client.predict(
	text="こんにちわ。",
	speaker="Hana",
	language="日本語",
	speed=1,
	api_name="/tts"
)
print(result)
