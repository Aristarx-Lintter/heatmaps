image_folder = "./images/"
target_question = """Describe the visible content of this image shortly, objectively and factually **as a single, continuous paragraph of prose.** Focus on the main subjects or elements present. Note their key visual characteristics (e.g., color, shape, texture if prominent) and their spatial arrangement or relationship to one another. If any actions or interactions are depicted, describe them simply.

**Crucially, do NOT use any structural formatting:**
*   Avoid bullet points, numbered lists, or any list-like structures.
*   Do not use introductory phrases that signal a list or structure (e.g., "Key elements include:", "The characteristics are:", "Present are:").
*   Do not use section headings or labels within the description.

Integrate all descriptive details smoothly into the narrative paragraph. Stick strictly to what is visually present. Avoid interpretations, assumptions, artistic analysis, or personal opinions. Make the description short.
"""
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
            {"type": "text", "text": target_question},
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
