from optimized_loader import Dataset
from persona import *

import textwrap
import re
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.tasks import Task
from camel.types import ModelPlatformType, ModelType
from camel.workforce import Workforce
import os

import pandas as pd



def create_judge_template(persona, example_feedback, criteria):
    msg_content_template = textwrap.dedent(
        f"""\
            You are a judge in a competition of writing Essays.
            This is your persona that you MUST act with: {persona}
            Here is an example that you might give with your persona, you MUST try your best to align with this:
            {example_feedback}
            When evaluating essays, you should consider the following criteria:
            {criteria}
            You also need to give score based on the criteria, from 1-6. The score given should be like 3/6, 5/6, 1/6, etc.

            Full text: {{full_text}}

            Assigned part: {{assigned_discourse}}

            Other parts: {{other_discourses}}
            """
    )
    
    # model = ModelFactory.create(
    #     model_platform=ModelPlatformType.OLLAMA,
    #     model_type="llama3",
    #     url="http://localhost:11434/v1",
    #     model_config_dict={"temperature": 0, "tools": None},
    # )
    model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,   
            model_config_dict={"temperature": 0}
        )
    
    return msg_content_template, model
def extract_discourses(essay, judge_role):
    discourse_parts = {key: val for key, val in essay.discourses.items() if val}
    
    assigned_discourse = discourse_parts.get(judge_role, "Not available")
    other_discourses = {k: v for k, v in discourse_parts.items() if k != judge_role}
    
    return assigned_discourse, other_discourses

def update_judge(agent, msg_template, full_text, assigned_discourse, other_discourses):
    updated_msg = msg_template.format(
        full_text=full_text,
        assigned_discourse=assigned_discourse,
        other_discourses=other_discourses
    )
    sys_msg = BaseMessage.make_assistant_message(
        role_name="Essay Judge",
        content=updated_msg
    )
    
    agent.system_message = sys_msg



def generate_task_content(essay):
    essay_info = (
        f"Evaluate the essay on topic '{essay.prompt_name}' written by a student in {essay.grade_level} Grade, "
        f"with ELL (English Language Learning) Status: {essay.ell_status}. "
    )

    task_content = (
        essay_info +
        "Each judge should evaluate the essay based on the specific criteria assigned to them. "
        "First, review the entire essay to understand its context. Then, each judge should focus on their designated section "
        "and give a score accordingly. Finally, list the opinions from each judge, making sure to preserve the unique identity "
        "of each judge, along with their score and name. "
        "Conclude with a final summary of the overall opinions and scores. "
        "Output should be structured as follows:\n\n"
        "<ScoresPerJudge>\n"
        "Position Paula (Judge): X/6\n"
        "Claim Clara (Judge): X/6\n"
        "Counterclaim Carl (Judge): X/6\n"
        "Rebuttal Robert (Judge): X/6\n"
        "Evidence Eva (Judge): X/6\n"
        "Summary Susan (Judge): X/6\n"
        "Unannotated Olivia (Judge): X/6\n"
        "</ScoresPerJudge>\n\n"
        "<OpinionsPerJudge>\n"
        "Position Paula (Judge): 'Opinion about position here'\n"
        "Claim Clara (Judge): 'Opinion about claim here'\n"
        "Counterclaim Carl (Judge): 'Opinion about counterclaim here'\n"
        "Rebuttal Robert (Judge): 'Opinion about rebuttal here'\n"
        "Evidence Eva (Judge): 'Opinion about evidence here'\n"
        "Summary Susan (Judge): 'Opinion about conclusion here'\n"
        "Unannotated Olivia (Judge): 'Opinion about unannotated sections here'\n"
        "</OpinionsPerJudge>\n\n"
        "<FinalSummary>\n"
        "Overall Score: X/6\n"
        "The summary about the essay and performance of the student.\n"
        "</FinalSummary>"
    )



    return task_content




def parse_judges_feedback(feedback: str):
    scores_dict = {}
    opinions_dict = {}
    final_score = None
    final_summary = ""
    

    scores_section = re.search(r'<ScoresPerJudge>(.*?)</ScoresPerJudge>', feedback, re.DOTALL)
    if scores_section:
        scores_text = scores_section.group(1)
        score_matches = re.findall(r'(\w+) \w+ \(Judge\): (\d+)/\d+', scores_text)
        scores_dict = {judge: int(score) for judge, score in score_matches}

    opinions_section = re.search(r'<OpinionsPerJudge>(.*?)</OpinionsPerJudge>', feedback, re.DOTALL)
    if opinions_section:
        opinions_text = opinions_section.group(1)
        opinion_matches = re.findall(r'(\w+) \w+ \(Judge\): \'([^\']*)\'', opinions_text)
        opinions_dict = {judge: opinion for judge, opinion in opinion_matches}

    final_summary_section = re.search(r'<FinalSummary>(.*?)</FinalSummary>', feedback, re.DOTALL)
    if final_summary_section:
        final_summary_text = final_summary_section.group(1)
        
        score_match = re.search(r'Overall Score: (\d+(\.\d+)?)/\d+', final_summary_text)
        if score_match:
            final_score = float(score_match.group(1))
        final_summary = final_summary_text.split('\n', 1)[-1].strip()

    return scores_dict, opinions_dict, final_score, final_summary



def main(num_essays, set_type, output_dir):
    #Part of the Dataset 
    essay_id_result = []
    full_text_result = []
    gt_essay_score_result = []
    title_result = []
    assignment_result = []
    grade_level_result = []
    
    #Part of the Prediction/Judgement
    scores_per_section_result = []
    opinions_per_section_result = []
    final_score_result = []
    final_summary_result = []
    new_final_score_result = []
    new_final_score_as_per_dataset_result = []
    
    
    
    
    Essays = Dataset(shuffle=True)
    position_template, position_model = create_judge_template(position_persona, position_example_feedback, position_criteria)
    claim_template, claim_model = create_judge_template(claim_persona, claim_example_feedback, claim_criteria)
    counterclaim_template, counterclaim_model = create_judge_template(counterclaim_persona, counterclaim_example_feedback, counterclaim_criteria)
    rebuttal_template, rebuttal_model = create_judge_template(rebuttal_persona, rebuttal_example_feedback, rebuttal_criteria)
    evidence_template, evidence_model = create_judge_template(evidence_persona, evidence_example_feedback, evidence_criteria)
    summary_template, summary_model = create_judge_template(concluding_summary_persona, concluding_summary_example_feedback, concluding_summary_criteria)
    unannotated_template, unannotated_model = create_judge_template(unannotated_persona, unannotated_example_feedback, unannotated_criteria)

    position_agent = ChatAgent(BaseMessage.make_assistant_message("", ""), model=position_model)
    claim_agent = ChatAgent(BaseMessage.make_assistant_message("", ""), model=claim_model)
    counterclaim_agent = ChatAgent(BaseMessage.make_assistant_message("", ""), model=counterclaim_model)
    rebuttal_agent = ChatAgent(BaseMessage.make_assistant_message("", ""), model=rebuttal_model)
    evidence_agent = ChatAgent(BaseMessage.make_assistant_message("", ""), model=evidence_model)
    summary_agent = ChatAgent(BaseMessage.make_assistant_message("", ""), model=summary_model)
    unannotated_agent = ChatAgent(BaseMessage.make_assistant_message("", ""), model=unannotated_model)
    '''
#     model = {
#     "model_platform": ModelPlatformType.OLLAMA,
#     "model_type": "llama3",  
#     "url": "http://localhost:11434/v1",  
#     "model_config_dict": {
#         "temperature": 0.7, 
#         "tools": [],  
#     }
# }


    # coordinator_agent_kwargs = {
    #     "model": ModelFactory.create(
    #         model_platform=ModelPlatformType.OLLAMA,
    #         model_type="llama3",
    #         url="http://localhost:11434/v1",
    #         model_config_dict={"temperature": 0, "tools": None}
    #     ),  
    # }
    
    # task_agent_kwargs = {
    #     "model": ModelFactory.create(
    #         model_platform=ModelPlatformType.OLLAMA,
    #         model_type="llama3",
    #         url="http://localhost:11434/v1",
    #         model_config_dict={"temperature": 0, "tools": None}
    #     ),  
    # }
    # coordinator_agent_kwargs = {
    #     "model": ModelFactory.create(
    #         model_platform=ModelPlatformType.OPENAI,
    #         model_type=ModelType.GPT_4O_MINI,
            
    #         model_config_dict={"temperature": 0, "tools": None}
    #     ),  
    # }
    
    # task_agent_kwargs = {
    #     "model": ModelFactory.create(
    #         model_platform=ModelPlatformType.OPENAI,
    #         model_type=ModelType.GPT_4O_MINI,
    #         model_config_dict={"temperature": 0, "tools": None}
    #     ),  
    # }
    '''
    workforce = Workforce(
        description="Essay Competition",
        # coordinator_agent_kwargs=coordinator_agent_kwargs,
        # task_agent_kwargs=task_agent_kwargs
    )
    
    workforce.add_single_agent_worker('Position Paula (Judge)', worker=position_agent)
    workforce.add_single_agent_worker('Claim Clara (Judge)', worker=claim_agent)
    workforce.add_single_agent_worker('Counterclaim Carl (Judge)', worker=counterclaim_agent)
    workforce.add_single_agent_worker('Rebuttal Robert (Judge)', worker=rebuttal_agent)
    workforce.add_single_agent_worker('Evidence Eva (Judge)', worker=evidence_agent)
    workforce.add_single_agent_worker('Summary Susan (Judge)', worker=summary_agent)
    workforce.add_single_agent_worker('Organizer Olivia (Helper)', worker=unannotated_agent)
    essay_count = 0
    outer_loop_flag = False
    for essay_batch in Essays:
        for essay in essay_batch:
            essay_count += 1
            full_text = essay.full_text

            position_discourse, position_other = extract_discourses(essay, 'Position')
            claim_discourse, claim_other = extract_discourses(essay, 'Claim')
            counterclaim_discourse, counterclaim_other = extract_discourses(essay, 'Counterclaim')
            rebuttal_discourse, rebuttal_other = extract_discourses(essay, 'Rebuttal')
            evidence_discourse, evidence_other = extract_discourses(essay, 'Evidence')
            summary_discourse, summary_other = extract_discourses(essay, 'Concluding Statement')
            unannotated_discourse, unannotated_other = extract_discourses(essay, 'Unannotated')

            
            
            update_judge(position_agent, position_template, full_text, position_discourse, position_other)
            update_judge(claim_agent, claim_template, full_text, claim_discourse, claim_other)
            update_judge(counterclaim_agent, counterclaim_template, full_text, counterclaim_discourse, counterclaim_other)
            update_judge(rebuttal_agent, rebuttal_template, full_text, rebuttal_discourse, rebuttal_other)
            update_judge(evidence_agent, evidence_template, full_text, evidence_discourse, evidence_other)
            update_judge(summary_agent, summary_template, full_text, summary_discourse, summary_other)
            update_judge(unannotated_agent, unannotated_template, full_text, unannotated_discourse, unannotated_other)

            
            task = Task(content=generate_task_content(essay), additional_info=full_text, id='0')
            result = workforce.process_task(task)
            scores_dict, opinions_dict, final_score, final_summary = parse_judges_feedback(result.result)
            print("------------------------------------------------------------------")
            print(f"Scores Per Section: {scores_dict}")
            print(f"Opinions Per Section: {opinions_dict}")
            print(f"Final Summary: {final_summary}")
            print(f"Final Score: {final_score} / 6")
            new_score = sum(score for section, score in scores_dict.items() if section != 'Unannotated')
            new_score += scores_dict['Unannotated'] / 2
            new_score_as_per_dataset = (new_score / 30) * 6
            print(f"New Final Score: {new_score} / 30")
            print(f"New Final Score as per dataset: {new_score_as_per_dataset} / 6")
        
            essay_id_result.append(essay.essay_id)
            full_text_result.append(essay.full_text)
            gt_essay_score_result.append(essay.holistic_essay_score)
            title_result.append(essay.prompt_name)
            assignment_result.append(essay.assignment)
            grade_level_result.append(essay.grade_level)
            
            scores_per_section_result.append(scores_dict)
            opinions_per_section_result.append(opinions_dict)
            final_score_result.append(final_score)
            final_summary_result.append(final_summary)
            new_final_score_result.append(new_score)
            new_final_score_as_per_dataset_result.append(new_score_as_per_dataset)
            if essay_count >= num_essays:
                outer_loop_flag = True
                break
        
        
        if outer_loop_flag:
            break
    
    
    #Saving the results in a csv file in the output directory mentioned in the argunments  
    data_dict = {
        'Essay ID': essay_id_result,
        'Full Text': full_text_result,
        'Ground Truth Essay Score': gt_essay_score_result,
        'Title': title_result,
        'Assignment': assignment_result,
        'Grade Level': grade_level_result,
        'Scores Per Section': scores_per_section_result,
        'Opinions Per Section': opinions_per_section_result,
        'Final Score': final_score_result,
        'Final Summary': final_summary_result,
        'New Final Score': new_final_score_result,
        'New Final Score as per Dataset': new_final_score_as_per_dataset_result,
    }

    df = pd.DataFrame(data_dict)
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"essays_count_{num_essays}_{current_time}.csv"
    df.to_csv(output_dir+output_filename, index=False)
    print(f"Output saved to {output_dir+output_filename}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_essays', type=int, default=1, help='Number of essays to evaluate')
    parser.add_argument("--set_type", type=str, default="train", help="Type of dataset to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/", help="Directory to save the output")
    args = parser.parse_args()
    num_essays = args.num_essays
    set_type = args.set_type
    output_dir = args.output_dir
    #The following got replaced by the above arguments from the command line
    #num_essays = 3
    # set_type = 'train'
    # output_dir = 'results/'

    main(num_essays, set_type, output_dir)