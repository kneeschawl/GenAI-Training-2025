import os
import argparse
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
import gradio as gr

# ------------------------------
# Utilities
# ------------------------------

DISCLAIMER = (
    "⚠️ Educational demo only. Not a substitute for professional diagnosis, "
    "treatment, or emergency help. If you’re in crisis, contact local emergency "
    "services or a suicide prevention helpline immediately."
)

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "self harm", "self-harm", "harm myself",
    "end my life", "ending my life", "want to die", "cut myself", "overdose"
]

def is_crisis(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in CRISIS_KEYWORDS)

def build_prompt(user_text: str) -> str:
    return (
        "You are a supportive, empathetic assistant for mental health conversations. "
        "Offer validation, gentle questions, and resources. Avoid medical advice.\n\n"
        f"User: {user_text}\n"
        "Therapist:"
    )

def detect_columns(df: pd.DataFrame):
    q_cols = ["question", "questionText", "Question", "question_title", "question_body"]
    a_cols = ["answer", "answerText", "Answer", "answer_body"]
    q_col = next((c for c in q_cols if c in df.columns), None)
    a_col = next((c for c in a_cols if c in df.columns), None)
    if q_col is None or a_col is None:
        raise ValueError(f"Could not find question/answer columns. Got columns: {list(df.columns)}.")
    return q_col, a_col

def format_row_to_text(question: str, answer: str) -> str:
    return f"User: {question.strip()}\nTherapist: {answer.strip()}\n"

# ------------------------------
# Data loading & tokenization
# ------------------------------

def load_counselchat_csv(csv_path: str) -> DatasetDict:
    df = pd.read_csv(csv_path)
    q_col, a_col = detect_columns(df)
    df = df.dropna(subset=[q_col, a_col])
    texts = [format_row_to_text(q, a) for q, a in zip(df[q_col].astype(str), df[a_col].astype(str))]
    ds = Dataset.from_dict({"text": texts})
    ds = ds.train_test_split(test_size=0.1, seed=42)
    return DatasetDict(train=ds["train"], validation=ds["test"])

def tokenize_function(examples, tokenizer, block_size: int):
    return tokenizer(examples["text"], truncation=True, max_length=block_size, padding="max_length")

# ------------------------------
# Training
# ------------------------------

def train(
    model_name: str,
    csv_path: str,
    output_dir: str,
    epochs: int = 3,
    per_device_batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 5e-5,
    block_size: int = 512,
    seed: int = 42,
):
    set_seed(seed)

    print(f"Loading dataset from: {csv_path}")
    ds_dict = load_counselchat_csv(csv_path)

    print(f"Loading tokenizer/model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    tokenized = ds_dict.map(lambda e: tokenize_function(e, tokenizer, block_size), batched=True, remove_columns=ds_dict["train"].column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    fp16 = torch.cuda.is_available()
    bf16 = False
    if not fp16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        bf16 = False

    # ------------------------------
    # Updated TrainingArguments (removed evaluation_strategy)
    # ------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_steps=500,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=fp16,
        bf16=bf16,
        report_to="none",
        logging_dir=os.path.join(output_dir, "logs"),
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training…")
    trainer.train()

    print("Saving model & tokenizer…")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")

# ------------------------------
# Inference / Gradio
# ------------------------------

class MHCChatbot:
    def __init__(self, model_dir: str, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        user_text: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        if is_crisis(user_text):
            return (
                f"{DISCLAIMER}\n\n"
                "It sounds like you might be going through a crisis. "
                "Please consider contacting your local emergency number, or a suicide prevention helpline.\n"
                "- 🇺🇸 US & Canada: 988\n"
                "- 🇬🇧 UK & ROI: 116 123\n"
                "- 🇮🇳 India: 91-22-27546669\n"
            )

        prompt = build_prompt(user_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "Therapist:" in text:
            return text.split("Therapist:", 1)[1].strip()
        return text.strip()

def launch_gradio(model_dir: str, server_port: int = 7860):
    bot = MHCChatbot(model_dir)

    def respond(user_input, max_new_tokens, temperature, top_p, top_k, repetition_penalty):
        if not user_input or not user_input.strip():
            return "Please enter your message."
        response = bot.generate(
            user_text=user_input.strip(),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
        )
        return f"{DISCLAIMER}\n\n{response}"

    with gr.Blocks(title="MHC GPT-2 (Demo)") as demo:
        gr.Markdown("# 🧠 MHC GPT-2 (Demo)\n" + DISCLAIMER)
        with gr.Row():
            user_in = gr.Textbox(label="Your message", placeholder="e.g., I feel anxious…", lines=4)
        with gr.Row():
            max_new_tokens = gr.Slider(16, 512, value=128, step=1, label="Max new tokens")
        with gr.Row():
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
            top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k")
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition penalty")
        out = gr.Textbox(label="Assistant", lines=10)
        go = gr.Button("Generate ✨")

        go.click(
            respond,
            inputs=[user_in, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
            outputs=[out],
        )

    demo.launch(server_name="0.0.0.0", server_port=server_port, show_api=False)

# ------------------------------
# CLI
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune GPT-2 on CounselChat and serve a Gradio demo.")
    p.add_argument("--csv_path", type=str, required=False)
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--output_dir", type=str, default="./mhc_gpt2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", action="store_true")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()

def main():
    args = parse_args()

    if args.train:
        if not args.csv_path:
            raise SystemExit("--csv_path is required for --train")
        train(
            model_name=args.model_name,
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            block_size=args.block_size,
            seed=args.seed,
        )

    if args.serve:
        if not os.path.exists(args.output_dir):
            raise SystemExit(f"Model dir not found: {args.output_dir}. Train first or point to an existing dir.")
        launch_gradio(args.output_dir, server_port=args.port)

    if not args.train and not args.serve:
        print("Nothing to do. Use --train and/or --serve. Run with -h for help.")

if __name__ == "__main__":
    main()
