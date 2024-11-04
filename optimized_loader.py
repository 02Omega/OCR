import pandas as pd
import numpy as np
import random
from tqdm import tqdm

class Essay:
    def __init__(
            self,
            essay_id=None,
            competition_set=None,
            full_text=None,
            holistic_essay_score=None,
            task=None,
            prompt_name=None,
            assignment=None,
            gender=None,
            grade_level=None,
            ell_status=None,
            race_ethnicity=None,
            economically_disadvantaged=None,
            student_disability_status=None,
            essay_word_count=None,
            discourses=None,
            discourses_effectiveness = None
    ):
        self.essay_id = essay_id
        self.full_text = full_text
        self.competition_set = competition_set
        self.holistic_essay_score = holistic_essay_score
        self.task = task
        self.prompt_name = prompt_name
        self.assignment = assignment
        self.gender = gender
        self.grade_level = grade_level
        self.race_ethnicity = race_ethnicity
        self.essay_word_count = essay_word_count
        self.ell_status = ell_status
        self.economically_disadvantaged = economically_disadvantaged if pd.notna(economically_disadvantaged) else "Unknown"
        self.student_disability_status = student_disability_status if pd.notna(student_disability_status) else "Unknown"
        self.discourses = discourses
        self.discourses_effectiveness = discourses_effectiveness

class Dataset:
    def __init__(self, PATH="PERSUADE/persuade_corpus_2.0.csv", batch_size=1, shuffle=False, set_type='train'):
        self.data = pd.read_csv(PATH, low_memory=False)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.set_type = set_type
        self.__init_dataset()

    def __init_dataset(self):
        
        filtered_data = self.data[self.data['competition_set'] == self.set_type]
        filtered_data['discourse_effectiveness'] = filtered_data['discourse_effectiveness'].fillna('Unknown')
        essays_dict = {}
        grouped = filtered_data.groupby('essay_id')
        discourse_types = ['Unannotated', 'Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement', 'Counterclaim', 'Rebuttal']
        for essay_id, group in tqdm(grouped, desc="Loading data"):
            f_values = group.iloc[0][['essay_id', 'competition_set', 'full_text', 'holistic_essay_score', 'task', 'prompt_name', 'assignment', 'gender', 'grade_level', 'ell_status', 'race_ethnicity', 'economically_disadvantaged', 'student_disability_status', 'essay_word_count']].to_dict()
            group_sorted = group.sort_values(by=['discourse_type', 'discourse_type_num'])
            
            discourses = {}
            effectiveness = {}
            for d_type in discourse_types:
                discourse_group = group_sorted[group_sorted['discourse_type'] == d_type]
                discourses[d_type] = list(discourse_group['discourse_text'])
                effectiveness[d_type] = list(discourse_group['discourse_effectiveness'])
            essays_dict[essay_id] = Essay(**f_values, discourses=discourses, discourses_effectiveness=effectiveness)
        
        self.essays = list(essays_dict.values())

    def __len__(self):
        return len(self.essays)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.essays)
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.essays):
            raise StopIteration
        batch = self.essays[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return batch

