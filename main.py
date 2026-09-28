import json
import os
import re

class AITrainingDataPipeline:
    def __init__(self, output_file="fine_tuning_dataset.jsonl"):
        self.output_file = output_file
        self.processed_count = 0
        self.rejected_count = 0

    def clean_text(self, text):
        """Standardizes whitespaces and trims text for AI training cleanliness."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def validate_quality(self, prompt, response):
        """Enforces high-quality data rules required to effectively train models."""
        # Rule 1: Eliminate empty or extremely short inputs
        if len(prompt) < 10 or len(response) < 10:
            return False, "Prompt or response text is too short."
        
        # Rule 2: Reject identical prompt-response combinations
        if prompt.lower() == response.lower():
            return False, "Prompt matches response (Looping vulnerability)."
            
        return True, "Passed"

    def format_for_llm(self, system_instruction, prompt, response):
        """Structures data into standard conversational format (ChatML schema)."""
        return {
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        }

    def process_raw_batch(self, system_prompt, raw_data_list):
        """Processes a batch of uncurated data items and writes them to disk."""
        print("Starting AI dataset optimization pipeline...\n")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for index, item in enumerate(raw_data_list):
                raw_p = item.get("raw_user_prompt", "")
                raw_r = item.get("raw_model_response", "")

                cleaned_p = self.clean_text(raw_p)
                cleaned_r = self.clean_text(raw_r)

                is_valid, reason = self.validate_quality(cleaned_p, cleaned_r)

                if is_valid:
                    formatted_entry = self.format_for_llm(system_prompt, cleaned_p, cleaned_r)
                    f.write(json.dumps(formatted_entry) + '\n')
                    self.processed_count += 1
                else:
                    print(f"[REJECTED ITEM {index}]: {reason}")
                    self.rejected_count += 1

        print(f"\nPipeline Finished!")
        print(f"Successfully Formatted: {self.processed_count} samples.")
        print(f"Flagged/Filtered Out: {self.rejected_count} samples.")


# --- DEMO INVOCATION FOR EVALUATION ---
if __name__ == "__main__":
    # Simulated system configuration for the AI target model
    SYSTEM_INSTRUCTIONS = "You are an advanced mathematical tutor that breaks down answers step-by-step."

    # Raw, dirty synthetic input dataset containing duplicates and bad formatting
    sample_dataset = [
        {
            "raw_user_prompt": "What   is  2  +   2?", 
            "raw_model_response": "2 + 2 equals 4. It is a basic arithmetic problem."
        },
        {
            "raw_user_prompt": "Too short", 
            "raw_model_response": "Short answer response."
        },
        {
            "raw_user_prompt": "Explain gravity.", 
            "raw_model_response": "Explain gravity."
        },
        {
            "raw_user_prompt": "Can you compute 10 times 5?", 
            "raw_model_response": "10 multiplied by 5 is exactly 50."
        }
    ]

    # Initialize and execute pipeline program
    pipeline = AITrainingDataPipeline()
    pipeline.process_raw_batch(SYSTEM_INSTRUCTIONS, sample_dataset)
