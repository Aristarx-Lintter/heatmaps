image_folder = "./images/"
messages_template = lambda image, transcription: [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_folder + image,
                "resized_height": 14 * 40,
                "resized_width": 14 * 80,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": transcription},
        ],
    }
]

answer_template = "<|im_start|>assistant\n{ans_text}<|im_end|>\n"
