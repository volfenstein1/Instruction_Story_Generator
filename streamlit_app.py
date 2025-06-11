import streamlit as st
import textstat
from transformers import pipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from trl import setup_chat_format

st.set_page_config(page_title="Story Generator")

st.title("Story Generator")

base_model = "SimpleStories/SimpleStories-35M"
peft_model = "volfenstein/LORA-simple-stories-generator-adapter"

# config = PeftConfig.from_pretrained(peft_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
model, tokenizer = setup_chat_format(model, tokenizer)

model = PeftModel.from_pretrained(model, peft_model)


def generate_story(topic, theme, wordcount, wordlength, complexity):
    user_prompt = """Topic: {topic}

    Theme: {theme}

    Wordcount: {wordcount}

    Wordlength: {wordlength}

    Complexity: {complexity}""".format(
        topic=topic,
        theme=theme,
        wordcount=wordcount,
        wordlength=wordlength,
        complexity=complexity,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    output = generator(
        [{"role": "user", "content": user_prompt}],
        max_new_tokens=512,
        return_full_text=False,
    )[0]

    return output["generated_text"]


def avg_word_length(text):
    punctuations = ".,!?;:'\"()-"
    for p in punctuations:
        text = text.replace(p, " ")

    # Split into words by whitespace
    words = text.split()

    if not words:
        return 0  # Avoid division by zero

    total_length = sum(len(word) for word in words)
    average_length = total_length / len(words)

    return average_length


themes = [
    "Family",
    "Deception",
    "Consciousness",
    "Growth",
    "Transformation",
    "Problem-Solving",
    "Magic",
    "Dreams",
    "Discovery",
    "Morality",
    "Coming of age",
    "Belonging",
    "Logic",
    "Celebration",
    "Planning",
    "Overcoming",
    "Friendship",
    "Honesty",
    "Helping Others",
    "Hardship",
    "The Five Senses",
    "Independence",
    "Amnesia",
    "Surprises",
    "Conscience",
    "Imagination",
    "Failure",
    "Agency",
    "Self-Acceptance",
    "Courage",
    "Hope",
    "Cooperation",
    "Humor",
    "Power",
    "Adventure",
    "Kindness",
    "Loss",
    "Strategy",
    "Curiosity",
    "Conflict",
    "Revenge",
    "Generosity",
    "Perseverance",
    "Scheming",
    "Travel",
    "Resilience",
    "Resourcefulness",
    "Teamwork",
    "Optimism",
    "Love",
]

topics = [
    "fantasy worlds",
    "hidden treasures",
    "magical objects",
    "royal kingdoms",
    "fairy tales",
    "the arts",
    "talking animals",
    "dream worlds",
    "riddles",
    "cultural traditions",
    "alien encounters",
    "subterranean worlds",
    "lost civilizations",
    "magical lands",
    "sports",
    "time travel",
    "haunted places",
    "gardens",
    "mystical creatures",
    "virtual worlds",
    "mysterious maps",
    "island adventures",
    "undercover missions",
    "unusual vehicles",
    "shape-shifting",
    "the sky",
    "school life",
    "invisibility",
    "robots and technology",
    "seasonal changes",
    "space exploration",
    "holidays",
    "sibling rivalry",
    "secret societies",
    "treasure hunts",
    "dinosaurs",
    "snowy adventures",
    "giant creatures",
    "a deadline or time limit",
    "pirates",
    "superheroes",
    "bygone eras",
    "outer space",
    "living objects",
    "lost cities",
    "enchanted forests",
    "underwater adventures",
    "miniature worlds",
]

left, right = st.columns(2, vertical_alignment="bottom")

selected_theme = left.selectbox(
    "Theme",
    themes,
    index=None,
    placeholder="Select a theme...",
)

selected_topic = right.selectbox(
    "Topic",
    topics,
    index=None,
    placeholder="Select a topic...",
)

selected_wordcount = st.slider("Target word count:", 100, 500, step=25)

selected_wordlength = st.slider(
    "Target Average Word Length", 3.2, 4.2, step=0.1
)

selected_complexity = st.slider("Complexity:", 0, 7)

submit = st.button("Generate")

if selected_theme and selected_topic and submit:
    with st.spinner("generating...", show_time=True):
        story = generate_story(
            topic=selected_topic,
            theme=selected_theme,
            wordcount=selected_wordcount,
            wordlength=selected_wordlength,
            complexity=selected_complexity,
        )
    st.write(story)
    st.write(
        "Word count:",
        len(story.split(" ")),
        "Average Word Length:",
        round(avg_word_length(story), 2),
        "Flesch Kincaid Grade:",
        round(textstat.flesch_kincaid_grade(story), 2),
    )
