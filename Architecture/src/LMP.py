from utils import path
import re

import requests

def chat_ollama(messages: list,temperature: float = 0.0):
    response = requests.post("http://134.100.39.10:30002/api/chat", json={
        "model": "llama3:70b",
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 512
        }
    })
    return response.json()["message"]["content"]


def prompt_ollama(prompt:str,temperature: float= 0.0):
    response = chat_ollama([{"role": "user", "content": prompt}],temperature)
    return response

def read_file(file_path):
    with open(file_path, "r") as file:
        prompt = file.read()
    return prompt


class LMP():
    def __init__(self,baseprompt):
        self.baseprompt = baseprompt
    def prompt(self,query):
        # "# Query: {query}\n objects = [{objects}]"    --> for planner_prompt
        # "# Check action: {action}"                    --> for check_prompt
        # "# Action: {action}"                          --> for action_prompt
        # TODO threshold set to 5cm as total distance check for that
        # TODO threshold for pushing in and pushing out in planner
        new_prompt = f"{self.baseprompt}\n{query}"
        user1 = f"I would like you to help me write Python code to control a gripper attached to a robot arm operating in a tabletop environment. Please complete the code every time when I give you new query. Pay attention to appeared patterns in the given context code. Be thorough and thoughtful in your code. Do not include any import statement. Do not repeat my question. Do not provide any text explanation (comment in code is okay). I will first give you the context of the code below:\n\n```\n{new_prompt}\n```\n\nNote that x is back to front, y is left to right, and z is bottom to up"
        assistant1 = f'Got it. I will complete what you give me next.'
        user2 = f'{query}'
        messages=[
                    {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a gripper attached to a robot arm in a tabletop environment."},
                    {"role": "user", "content": user1},
                    {"role": "assistant", "content": assistant1},
                    {"role": "user", "content": user2},
                ]
        for message in messages:
            print("Role: ",message["role"])
            print("Content: ",message["content"])
            print("\n")
        raise
        response = chat_ollama(messages)
        return response
    
def hierarchial_code_generation(task_description: str,objects: list)->str:
    """hierachally generate LMP based on task_description and a list of object names""" 
    # Define LMPs
    planner_baseprompt = read_file(path+"prompts/planner_prompt.py")
    planner_lmp = LMP(planner_baseprompt)
    check_baseprompt = read_file(path+"prompts/check_prompt.py")
    check_lmp = LMP(check_baseprompt)
    action_baseprompt = read_file(path+"prompts/action_prompt.py")
    action_lmp = LMP(action_baseprompt)

    # Query planner_lmp
    planner_query = "# Query: "+ task_description + "\nobjects = "+ str(objects)
    planner_response = planner_lmp.prompt(planner_query)
    answer = planner_query+"\n"+"# Call Planner LMP: planner(\""+ task_description + "\")\n"
    # Hierarchically call check_lmp and action_lmp if composer function called 
    for line in planner_response.split("\n"):
        if line.strip().startswith("composer("):
            # hierarchical call defined to call check and then action
            composer_pattern = r'composer\("([^"]+)"\)'
            composer_action = re.findall(composer_pattern, line)[0]
            # First do action and then check if action changed state
            # add intendation and mark from line with composer call that is replaced 
            action_query = "# Action: "+ composer_action
            action_response = action_lmp.prompt(action_query)
            action_response = "    " + "# Call Action LMP: " + line.strip() +"\n    " + "\n    ".join(action_response.split("\n"))
            answer += action_response+"\n"
            # add intendation and mark from line with composer call that is replaced 
            check_query = "# Check action: "+ composer_action
            check_response = check_lmp.prompt(check_query)
            check_response = "    " + "# Call Check LMP: " + line.strip() +"\n    " + "\n    ".join(check_response.split("\n"))
            answer += check_response+"\n"
        else:
            answer += line + "\n"
    return answer

if __name__ == "__main__":
    # answer = hierarchial_code_generation("Unplug charger",['Charger'])
    # print(answer)
    LMP = LMP("BASEPROMPT").prompt("TASK DESCRIPTION")