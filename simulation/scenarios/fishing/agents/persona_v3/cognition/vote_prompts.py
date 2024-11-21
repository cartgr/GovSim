from datetime import datetime
from typing import Optional

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, user

from .utils import (
    get_sytem_prompt,
    location_time_info,
    memory_prompt,
    reasoning_steps_prompt,
)


def prompt_voting_decision(
    model: ModelWandbWrapper,
    identity: PersonaIdentity,
    memories: list[str],
    current_location: str,
    current_time: datetime,
    context: str,
    harvesting_data: dict[str, int],
    other_agents: list[str],
) -> tuple[Optional[str], str, str]:
    """
    Prompt the agent to make a voting decision about suspending another agent.

    Returns:
        tuple: (vote_for, reason, html_interactions)
            vote_for: Name of agent to vote for suspension, or None for no vote
            reason: Explanation for the vote
            html_interactions: HTML formatted interaction log
    """
    # Special handling for Jack
    if identity.name.lower() == "jack":
        print("\n=== VOTING PHASE ===")
        # Print out the system prompt and context like the model would see
        print("\n=== PROMPT for reasoning ===")
        print(f"{get_sytem_prompt(identity)}")
        print(location_time_info(current_location, current_time))
        print(memory_prompt(identity, memories))
        print(f"\nVoting Phase Information:")
        print(f"- Monthly harvest report: {harvesting_data}")
        print(f"- Other agents: {', '.join(other_agents)}")
        print(
            "\nTask: Decide if you want to vote to suspend any agent from fishing for the next round only."
        )
        print("Consider:")
        print("- The sustainability of the shared resource")
        print("- Each agent's harvesting behavior")
        print("- Impact on the community")
        print("- Whether a one-round suspension is warranted")
        print("==================\n")

        while True:
            vote = input("Who would you like to vote to suspend? ").strip()
            if vote.lower() == "none" or vote in other_agents:
                break
            print(f"Please enter a valid name from {other_agents} or 'none'")

        reason = input("Enter the reason for your vote: ")

        # Format response similar to model output
        html_response = (
            f"<strong>Vote Decision</strong><br>"
            f"Vote: {vote}<br>"
            f"Reason: {reason}"
        )

        vote_for = None if vote.lower() == "none" else vote
        return vote_for, reason, html_response

    # Modified model logic for other personas
    lm = model.start_chain(identity.name, "voting_cognition", "choose_vote")

    with user():
        lm += f"{get_sytem_prompt(identity)}\n"
        lm += location_time_info(current_location, current_time)
        lm += memory_prompt(identity, memories)
        lm += f"\nVoting Phase Information:"
        lm += f"\n- Monthly harvest report: {harvesting_data}"
        lm += f"\n- Other agents: {', '.join(other_agents)}"
        lm += f"\n\nTask: Decide if you want to vote to suspend any agent from fishing for the next round only."
        lm += f"\nConsider:"
        lm += f"\n- The sustainability of the shared resource"
        lm += f"\n- Each agent's harvesting behavior"
        lm += f"\n- Impact on the community"
        lm += f"\n- Whether a one-round suspension is warranted"
        lm += reasoning_steps_prompt()
        lm += "\n\nAfter your analysis, you must end your response with exactly two lines in this format:"
        lm += "\nReason: [your clear and concise reason for voting]"
        lm += "\nVote: [exactly one of: none, " + ", ".join(other_agents) + "]"

    with assistant():
        lm = model.gen(lm, "reasoning")
        response = lm.text_to_consume

        # Get the last few lines where the Reason and Vote should be
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        reason_line = None
        vote_line = None

        # Look for the last occurrence of Reason: and Vote:
        for line in reversed(lines):
            if line.startswith("Reason:") and not reason_line:
                reason_line = line
            elif line.startswith("Vote:") and not vote_line:
                vote_line = line
            if reason_line and vote_line:
                break

        # Extract reason and vote with validation
        reason = reason_line[7:].strip() if reason_line else "No reason provided"
        vote = vote_line[5:].strip() if vote_line else "none"

        # Validate vote is one of the allowed options
        allowed_votes = ["none"] + other_agents
        if vote.lower() not in [v.lower() for v in allowed_votes]:
            vote = "none"

        # Add the parsed values to the chain using proper methods
        lm = model.find(
            lm,
            regex=r"Reason:\s*(.+?)(?=\n|Vote:|$)",
            default_value=reason,
            name="reason",
        )
        lm = model.find(
            lm, regex=r"Vote:\s*(.+?)(?=\n|$)", default_value=vote, name="vote"
        )

    model.end_chain(identity.name, lm)

    # Convert "none" to None for no vote
    vote_for = None if vote.lower() == "none" else vote
    return vote_for, reason, lm.html()
