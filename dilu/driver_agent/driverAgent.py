import os
import textwrap
import time
from rich import print
from typing import List

# UPDATED IMPORTS
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback, OpenAICallbackHandler
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from dilu.scenario.envScenario import EnvScenario

delimiter = "####"
# ... (Keep example_message and example_answer variables as they are in original) ...
example_message = textwrap.dedent(f"""\
        {delimiter} Driving scenario description:
        You are driving on a road with 4 lanes, and you are currently driving in the second lane from the left. Your speed is 25.00 m/s, acceleration is 0.00 m/s^2, and lane position is 363.14 m. 
        There are other vehicles driving around you, and below is their basic information:
        - Vehicle `912` is driving on the same lane of you and is ahead of you. The speed of it is 23.30 m/s, acceleration is 0.00 m/s^2, and lane position is 382.33 m.
        - Vehicle `864` is driving on the lane to your right and is ahead of you. The speed of it is 21.30 m/s, acceleration is 0.00 m/s^2, and lane position is 373.74 m.
        - Vehicle `488` is driving on the lane to your left and is ahead of you. The speed of it is 23.61 $m/s$, acceleration is 0.00 $m/s^2$, and lane position is 368.75 $m$.

        {delimiter} Your available actions:
        IDLE - remain in the current lane with current speed Action_id: 1
        Turn-left - change lane to the left of the current lane Action_id: 0
        Turn-right - change lane to the right of the current lane Action_id: 2
        Acceleration - accelerate the vehicle Action_id: 3
        Deceleration - decelerate the vehicle Action_id: 4
        """)
example_answer = textwrap.dedent(f"""\
        **Step-by-Step Explanation:**
        1. **Safety Check:** The vehicle directly ahead in my lane (Vehicle 912) is only 19.19 meters away (382.33m - 363.14m), which is under the 25m safety buffer. It is also traveling slower than me (23.30 m/s vs 25.00 m/s). This presents an immediate collision risk.
        2. **Efficiency Consideration:** My current speed is 25.00 m/s, which is close to the 28 m/s target, but safety supersedes efficiency. Accelerating or maintaining speed will cause a rear-end collision.
        3. **Lane Change Feasibility:** Changing lanes is not safe. The left lane is blocked by Vehicle 488 (only 5.61m ahead), and the right lane is blocked by Vehicle 864 (only 10.6m ahead). Both are too close to attempt a safe lane change.
        4. **Conclusion:** Since I cannot safely maintain speed, accelerate, or change lanes due to surrounding traffic, I must decelerate to avoid crashing into Vehicle 912.

        **Answer:**
        Reasoning: The lead car in my lane is critically close (under 25m) and slower, and adjacent lanes are blocked, mandating immediate deceleration.
        Response to user:{delimiter} 4
        """)


class DriverAgent:
    def __init__(
            self, sce: EnvScenario,
            temperature: float = 0, verbose: bool = False
    ) -> None:
        self.sce = sce
        oai_api_type = os.getenv("OPENAI_API_TYPE")
        if oai_api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                callbacks=[
                    OpenAICallbackHandler()
                ],
                deployment_name=os.getenv("AZURE_CHAT_DEPLOY_NAME"),
                temperature=temperature,
                max_tokens=2000,
                request_timeout=60,
                streaming=True,
            )
        elif oai_api_type == "openai":
            print("Use OpenAI API")
            self.llm = ChatOpenAI(
                temperature=temperature,
                callbacks=[
                    OpenAICallbackHandler()
                ],
                model_name=os.getenv("OPENAI_CHAT_MODEL"),
                max_tokens=2000,
                request_timeout=60,
                streaming=True,
            )
        # [ADD] Added support for local Ollama models
        elif oai_api_type == "ollama":
            model_name = os.getenv("OLLAMA_CHAT_MODEL")
            api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
            api_key = os.getenv("OLLAMA_API_KEY", "ollama")

            print(f"Using Local Ollama API: {model_name} at {api_base}")

            # Use ChatOpenAI to talk to Ollama's OpenAI-compatible endpoint
            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name=model_name,
                openai_api_base=api_base,  # or base_url for newer langchain versions
                openai_api_key=api_key,
                max_tokens=2000,
                request_timeout=60,
                streaming=True,
            )
        else:
            raise ValueError(f"Unknown OPENAI_API_TYPE: {oai_api_type}")

    def few_shot_decision(self, scenario_description: str = "Not available", previous_decisions: str = "Not available",
                          available_actions: str = "Not available", driving_intensions: str = "Not available",
                          fewshot_messages: List[str] = None, fewshot_answers: List[str] = None):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/

        system_message = textwrap.dedent(f"""\
        You are an autonomous driving decision module. You must strictly follow a Chain of Thought reasoning process before making a decision.

        ### DRIVE LOGIC:
        1. SAFETY: If a lead car is closer than 25m and your speed is higher, you MUST decelerate (Action_id 4).
        2. EFFICIENCY: Maintain a target speed of 28m/s. Change lanes (0 or 2) if blocked by slower traffic.

        You MUST format your response EXACTLY like this, using these exact headings:

        **Step-by-Step Explanation:**
        1. **Safety Check:** [Analyze distance and speed of the lead car and adjacent cars]
        2. **Efficiency Consideration:** [Analyze if you need to speed up to reach the target speed]
        3. **Lane Change Feasibility:** [Analyze if changing lanes is safe or necessary]
        4. **Conclusion:** [Summarize the best course of action]

        **Answer:**
        Reasoning: [1-sentence summary of the conclusion]
        Response to user:{delimiter} <Action_id_integer>
        """)

        human_message = f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Driving Intensions:
        {driving_intensions}
        {delimiter} Available actions:
        {available_actions}

        You can stop reasoning once you have a valid action to take. 
        """
        human_message = human_message.replace("        ", "")

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),
            # HumanMessage(content=example_message),
            # AIMessage(content=example_answer),
        ]
        for i in range(len(fewshot_messages)):
            messages.append(
                HumanMessage(content=fewshot_messages[i])
            )
            messages.append(
                AIMessage(content=fewshot_answers[i])
            )
        messages.append(
            HumanMessage(content=human_message)
        )
        # print("fewshot number:", (len(messages) - 2)/2)
        start_time = time.time()

        # NOTE: get_openai_callback might return 0 for Ollama
        # with get_openai_callback() as cb:
        # response = self.llm.invoke(messages) # invoke instead of __call__

        print("[cyan]Agent answer:[/cyan]")
        response_content = ""
        # .stream() is widely supported in newer langchain versions
        for chunk in self.llm.stream(messages):
            response_content += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")
        decision_action = response_content.split(delimiter)[-1]
        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            print("Output is not a int number, checking the output...")
            check_message = f"""
            You are a output checking assistant who is responsible for checking the output of another agent.

            The output you received is: {decision_action}

            Your should just output the right int type of action_id, with no other characters or delimiters.
            i.e. :
            | Action_id | Action Description                                     |
            |--------|--------------------------------------------------------|
            | 0      | Turn-left: change lane to the left of the current lane |
            | 1      | IDLE: remain in the current lane with current speed   |
            | 2      | Turn-right: change lane to the right of the current lane|
            | 3      | Acceleration: accelerate the vehicle                 |
            | 4      | Deceleration: decelerate the vehicle                 |


            You answer format would be:
            {delimiter} <correct action_id within 0-4>
            """
            messages = [
                HumanMessage(content=check_message),
            ]
            with get_openai_callback() as cb:
                check_response = self.llm.invoke(messages)  # Changed from self.llm(messages)
            result = int(check_response.content.split(delimiter)[-1])

        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + \
                                      "\n---------------\n"
        print("Result:", result)
        return result, response_content, human_message, few_shot_answers_store