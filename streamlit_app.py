import streamlit as st
import textstat
from transformers import pipeline
import torch

st.set_page_config(page_title="Story Generator")

st.title("Story Generator")


@st.cache_resource
def generate_story(topic, theme, wordcount, paragraphs, complexity):
    user_prompt = """Topic: {topic}

    Theme: {theme}

    Wordcount: {wordcount}

    Paragraphs: {paragraphs}

    Complexity: {complexity}""".format(
        topic=topic,
        theme=theme,
        wordcount=wordcount,
        paragraphs=paragraphs,
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
        model="volfenstein/wolfgang-story-generator",
        device=device,
    )
    output = generator(
        [{"role": "user", "content": user_prompt}],
        max_new_tokens=128,
        return_full_text=False,
    )[0]

    return output["generated_text"]


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

selected_wordcount = st.slider("Target word count:", 50, 800, step=25)

selected_paragraphs = st.slider("Number of paragraphs", 1, 9)

selected_complexity = st.slider("Complexity:", 0, 12)

submit = st.button("Generate")

if selected_theme and selected_topic and submit:
    with st.spinner("generating...", show_time=True):
        story = generate_story(
            topic=selected_topic,
            theme=selected_theme,
            wordcount=selected_wordcount // 25,
            paragraphs=selected_paragraphs,
            complexity=selected_complexity,
        )
    st.write(story)
    st.write(
        "Word count:",
        len(story.split(" ")),
        "Paragraphs:",
        len(story.split("\n")),
        "Flesch Kincaid Grade:",
        round(textstat.flesch_kincaid_grade(story), 1),
    )
