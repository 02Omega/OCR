from camel.agents.chat_agent import ChatAgent
from camel.configs.openai_config import ChatGPTConfig
from camel.messages.base import BaseMessage
from camel.models import ModelFactory
from camel.tasks.task import Task
from camel.toolkits import SEARCH_FUNCS, WEATHER_FUNCS, GoogleMapsToolkit
from camel.types import ModelPlatformType, ModelType
from camel.workforce import Workforce
import os
from getpass import getpass




if __name__ == '__main__':
    #Setting up API KEY
    openai_api_key = getpass("API Key: ")
    os.environ["OPENAI_API_KEY"] = openai_api_key


    #Single Worker Agent 1, Guiding everyone on the tour to have fun
    guide_sysmsg = BaseMessage.make_assistant_message(
        role_name="tour guide",
        content = "You have to lead everyone to have fun"
    )
    #Single Worker Agent 2, Planning the tour
    planner_sysmsg = BaseMessage.make_assistant_message(
        role_name="planner",
        content = "good at tour plan"
    )
    #Create Chat Agents for the two workers above
    guide_agent = ChatAgent(guide_sysmsg)
    planner_agent = ChatAgent(planner_sysmsg)

    # Tools for the assistants to use
    function_list = [
        *SEARCH_FUNCS, # Importing all the search functions
        *WEATHER_FUNCS, # Importing all the weather functions
        *GoogleMapsToolkit().get_tools(), # Importing all the Google Maps functions
    ]

    #Setting up the model to be used for the agents in the workforce
    model_platform = ModelPlatformType.OPENAI
    model_type = ModelType.GPT_4O_MINI
    assistant_role_name = 'Searcher'
    user_role_name = 'Professor'
    #Setting up assistant model
    assistant_agent_kwargs = dict(
        model = ModelFactory.create(
            model_platform = model_platform,
            model_type = model_type,
            model_config_dict = ChatGPTConfig(temperature=0.0).as_dict()
        ),
        tools = function_list
    )
    # Setting up the user model 
    user_agent_kwargs = dict(
        model = ModelFactory.create(
            model_platform = model_platform,
            model_type = model_type,
            model_config_dict = ChatGPTConfig(temperature=0.0).as_dict()
        ),
    )
    #Creating a workforce "A travel group" with the roles and workers for each task
    workforce = Workforce('A travel group')
    workforce.add_role_playing_worker(
        'research Group',
        assistant_role_name,
        user_role_name,
        assistant_agent_kwargs,
        user_agent_kwargs,
        1
    ).add_single_agent_worker('tour guide', guide_agent).add_single_agent_worker('planner', planner_agent)

    #Human Task to be processed by the workforce
    humantask = Task(
        content="research history of Paris and plan a tour.",
        id='0'
    )
    #Processing the task by the workforce
    task = workforce.process_task(humantask)
    print('final result of original task:', task.result)