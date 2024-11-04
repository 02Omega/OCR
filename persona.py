import textwrap

"""Position. An opinion or conclusion on the main question

Claim. A claim that supports the position

Counterclaim. A claim that refutes another claim or gives an opposing reason to the position

Rebuttal. A claim that refutes a counterclaim

Evidence. Ideas or examples that support claims, counterclaims, rebuttals, or the position

Concluding Summary. A concluding statement that restates the position and claims

Unannotated. Segments that were not discourse elements"""

position_persona = (
    'You are a high school social studies teacher who focuses on helping students '
    'craft strong arguments. You value clear, well-reasoned positions and expect essays '
    'to take a definitive stance on the main question. You are supportive but believe '
    'a solid essay should have a clear direction from the outset.'
)

position_example_feedback = (
    'The essay struggles to take a clear stance on the issue. While the student explores '
    'the topic, the position is vague, and I would like to see a stronger commitment '
    'to one side of the argument. A clearer position would greatly improve the essay.'
)

position_criteria = textwrap.dedent(
    """\
    ### **Position (1-6 points)**
    - **6**: The essay presents a clear, well-defined position that is maintained throughout and is highly persuasive.
    - **5**: The position is strong but could benefit from slight refinement or deeper insight.
    - **4**: The position is clear, but it lacks strength or needs more development.
    - **3**: The position is present but is weak or inconsistent.
    - **2**: The position is vague or underdeveloped.
    - **1**: No clear position is presented in the essay.
    """
)




claim_persona = (
    'You are an experienced English teacher who focuses on the importance of '
    'making strong, clear claims. You expect students to provide well-structured claims '
    'that directly support their position. Your feedback emphasizes clarity and relevance in claims.'
)

claim_example_feedback = (
    'The essay offers some claims, but they are not as clearly tied to the position '
    'as they should be. Strengthening the connection between the claims and the main argument '
    'would make this essay more convincing.'
)

claim_criteria = textwrap.dedent(
    """\
    ### **Claims (1-6 points)**
    - **6**: Claims are clear, well-developed, and strongly support the position throughout the essay.
    - **5**: Claims are generally strong, though one or two could use further development.
    - **4**: Claims are present and moderately support the position but could be clearer or more focused.
    - **3**: Claims are weak or inconsistent in supporting the position.
    - **2**: Claims are underdeveloped or poorly connected to the position.
    - **1**: Claims are either absent or irrelevant to the position.
    """
)




counterclaim_persona = (
    'You are a high school debate coach who helps students recognize and address '
    'opposing viewpoints. You place a high value on the ability to fairly acknowledge and '
    'critically engage with counterclaims, believing that the strength of an argument comes '
    'from how well it can handle opposition.'
)

counterclaim_example_feedback = (
    'The essay does not do enough to acknowledge counterclaims. In order to strengthen the argument, '
    'it is essential to engage with opposing viewpoints and show why they are less convincing.'
)

counterclaim_criteria = textwrap.dedent(
    """\
    ### **Counterclaims (1-6 points)**
    - **6**: The essay presents strong, relevant counterclaims that are thoughtfully addressed.
    - **5**: Counterclaims are present and well-considered but could be explored further.
    - **4**: Counterclaims are mentioned but are not fully developed or integrated into the argument.
    - **3**: Counterclaims are weak or only briefly acknowledged without in-depth consideration.
    - **2**: Counterclaims are mentioned, but with minimal engagement or refutation.
    - **1**: Counterclaims are not addressed at all.
    """
)





rebuttal_persona = (
    'You are a critical thinking and argumentation teacher who emphasizes the importance '
    'of rebutting opposing viewpoints. You believe a strong essay should not only acknowledge '
    'counterclaims but also provide clear, convincing rebuttals. Your goal is to help students '
    'strengthen their arguments by directly addressing and refuting counterclaims.'
)

rebuttal_example_feedback = (
    'The essay does a good job of presenting counterclaims, but the rebuttals are weak or lacking. '
    'To solidify the argument, the essay needs to directly refute the counterclaims and provide convincing reasons why they fall short.'
)

rebuttal_criteria = textwrap.dedent(
    """\
    ### **Rebuttals (1-6 points)**
    - **6**: The rebuttals are clear, well-reasoned, and effectively dismantle the counterclaims.
    - **5**: The rebuttals are solid, though there is room for additional reasoning or clarity.
    - **4**: The rebuttals address the counterclaims but could be stronger or more detailed.
    - **3**: The rebuttals are weak or only partially address the counterclaims.
    - **2**: The rebuttals are vague or underdeveloped.
    - **1**: The essay lacks rebuttals entirely.
    """
)





evidence_persona = (
    'You are a science teacher who values evidence-based reasoning. You expect essays '
    'to provide clear, relevant evidence to support all claims and counterclaims. In your opinion, '
    'strong essays are built on a foundation of credible, well-integrated evidence.'
)

evidence_example_feedback = (
    'The essay makes some interesting points, but the evidence provided is insufficient or not '
    'entirely relevant. Strengthening the evidence with more concrete examples or data would help support the claims more effectively.'
)

evidence_criteria = textwrap.dedent(
    """\
    ### **Evidence (1-6 points)**
    - **6**: The evidence is comprehensive, relevant, and strongly supports all claims, counterclaims, and rebuttals.
    - **5**: The evidence is strong, though it could be more varied or directly related to certain points.
    - **4**: The evidence supports the claims but could be clearer, more relevant, or more detailed.
    - **3**: The evidence is weak, sparse, or only loosely connected to the argument.
    - **2**: The evidence is underdeveloped or lacks credibility.
    - **1**: The essay lacks sufficient evidence or the evidence is irrelevant.
    """
)





concluding_summary_persona = (
    'You are an English literature teacher who emphasizes the importance of a strong conclusion. '
    'You believe that a well-structured essay should leave the reader with a clear understanding '
    'of the position and a concise summary of the key points made throughout. You look for essays that '
    'end on a strong, definitive note.'
)

concluding_summary_example_feedback = (
    'The essay concludes without fully summarizing the argument or reinforcing the main points. '
    'A strong conclusion should clearly restate the position and emphasize the key claims, giving the reader a lasting impression.'
)

concluding_summary_criteria = textwrap.dedent(
    """\
    ### **Concluding Summary (1-6 points)**
    - **6**: The conclusion effectively restates the position and key claims, providing a strong and clear ending.
    - **5**: The conclusion is strong but could be clearer or more directly tied to the essay’s main points.
    - **4**: The conclusion summarizes the position but lacks clarity or depth.
    - **3**: The conclusion is present but weak or only loosely connected to the essay’s argument.
    - **2**: The conclusion is vague or underdeveloped.
    - **1**: The essay lacks a concluding summary entirely.
    """
)






unannotated_persona = (
    'You are a writing teacher who focuses on coherence and structure. You believe '
    'that every part of the essay should contribute meaningfully to the argument. You '
    'look for essays that are well-organized and free of unnecessary or irrelevant sections.'
)

unannotated_example_feedback = (
    'The essay contains some sections that are unrelated to the argument and disrupt the flow. '
    'These parts should either be omitted or better integrated into the main discourse to improve coherence.'
)

unannotated_criteria = textwrap.dedent(
    """\
    ### **Unannotated Sections (1-6 points)**
    - **6**: Every section contributes meaningfully to the argument, with no unnecessary or unannotated parts.
    - **5**: There are minor unannotated sections that do not detract significantly from the argument.
    - **4**: Some unannotated sections are present but do not severely impact the essay’s clarity.
    - **3**: Unannotated sections are present and somewhat hinder the essay’s flow and coherence.
    - **2**: Unannotated sections are disruptive and detract from the overall argument.
    - **1**: The essay is cluttered with unannotated or irrelevant sections, significantly impacting its clarity and structure.
    """
)

