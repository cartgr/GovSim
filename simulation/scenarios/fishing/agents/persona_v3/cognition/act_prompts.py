from datetime import datetime

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .utils import (
    consider_identity_persona_prompt,
    conversation_to_string_with_dash,
    get_sytem_prompt,
    list_to_comma_string,
    list_to_string_with_dash,
    location_time_info,
    memory_prompt,
    reasoning_steps_prompt,
)


def prompt_action_choose_amount_of_fish_to_catch(
    model: ModelWandbWrapper,
    identity: PersonaIdentity,
    memories: list[str],
    current_location: str,
    current_time: datetime,
    context: str,
    interval: list[int],
    consider_identity_persona: bool = True,
):
    # Add check for Jack AND check if suspended
    if identity.name.lower() == "jack":
        # First check if Jack is suspended
        if "SUSPENDED" in context or any(
            "suspended" in str(memory).lower() for memory in memories
        ):
            print("\n=== SUSPENSION NOTICE ===")
            print("Jack is currently suspended from fishing this round.")
            return 0, "<strong>Status:</strong> SUSPENDED"

        print(f"\nFishing range: {interval[0]}-{interval[-1]} tons")
        # Print out the system prompt and context like the model would see
        print("\n=== PROMPT for reasoning ===")
        print(f"{get_sytem_prompt(identity)}")
        print(location_time_info(current_location, current_time))
        print(memory_prompt(identity, memories))
        print(
            f"\nTask: With a fishing range set between {interval[0]}-{interval[-1]}, how many tons of fish would you catch this month? "
        )
        print(reasoning_steps_prompt())
        print('Put the final answer after "Answer:", example Answer: N tons.')
        print("==================\n")
        reasoning = input("Enter Jack's reasoning for fishing amount: ")
        while True:
            try:
                option = int(input("Enter amount of fish Jack will catch (in tons): "))
                if interval[0] <= option <= interval[-1]:
                    break
                print(f"Amount must be between {interval[0]} and {interval[-1]} tons")
            except ValueError:
                print("Please enter a valid number")

        # Format the response similar to the model output
        html_response = f"<strong>Reasoning:</strong> {reasoning}<br><strong>Answer:</strong> {option} tons"
        return option, html_response

    # Original model logic for other personas
    lm = model.start_chain(identity.name, "fishing_cognition_act", "choose_act_options")

    with user():
        lm += f"{get_sytem_prompt(identity)}\n"
        lm += location_time_info(current_location, current_time)
        lm += memory_prompt(identity, memories)
        lm += f"\n"
        lm += f"Task: With a fishing range set between {interval[0]}-{interval[-1]}, how many tons of fish would you catch this month? "
        lm += reasoning_steps_prompt()
        lm += ' Put the final answer after "Answer:", example Answer: N tons.'

    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=r"Answer:|So, the answer is:|\*\*Answer\*\*:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            stop_regex=f"tons",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(identity.name, lm)

    return option, lm.html()
