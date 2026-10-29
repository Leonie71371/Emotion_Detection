EMOTIONS = [
        "anger",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise"
    ]

class Prompting:
    
    def __init__(self) -> None:
        pass

    def labels_to_text(self, record: dict) -> str:
        kept = [emotion for emotion in EMOTIONS if record[emotion] == 1]

        if not kept:
            label_text = "none"
        else:
            label_text = ", ".join(kept)

        return label_text  

    def format_example(self, record: dict) -> str:
        label_text = self.labels_to_text(record)
        block = f"Text: {record['text']}\nEmotions: {label_text}"

        return block
    
    def build_message(self, test_text: str, examples: list[dict]) -> list[dict]:
        system_message = (
            "Given the following sentence, identify the emotions expressed.\nPossible emotions are: "
            + ", ".join(EMOTIONS) + "."
            + "\nThe answer should only contain the emotion names, comma-separated in the order they were listed, or 'none' if there is no emotion detected."
        )

        user_prompt = ""
        
        for example in examples:
            user_prompt += self.format_example(example) + "\n\n"

        user_prompt += f"\nTest_Text: {test_text}. Emotions detected: "

        message = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ]

        return message