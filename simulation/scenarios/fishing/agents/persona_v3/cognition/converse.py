from datetime import datetime

from simulation.persona.cognition.converse import ConverseComponent
from simulation.persona.cognition.retrieve import RetrieveComponent
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .converse_prompts import (
    prompt_converse_utterance_in_group,
    prompt_summarize_conversation_in_one_sentence,
)
from .reflect_prompts import prompt_find_harvesting_limit_from_conversation


class FishingConverseComponent(ConverseComponent):
    def __init__(
        self,
        model: ModelWandbWrapper,
        retrieve: RetrieveComponent,
        cfg,
    ):
        super().__init__(model, retrieve, cfg)
        self.fishing_reports = []  # Store historical reports

    def converse_group(
        self,
        target_personas: list[PersonaIdentity],
        current_location: str,
        current_time: datetime,
        current_context: str,
        agent_resource_num: dict[str, int],
        lake_fish_before: int,
        lake_fish_after: int,
    ) -> tuple[list[tuple[str, str]], str]:
        current_conversation: list[tuple[PersonaIdentity, str]] = []
        html_interactions = []

        # Inject fake conversation about how many fish each person caught
        if (
            self.cfg.inject_resource_observation
            and self.cfg.inject_resource_observation_strategy == "individual"
        ):
            for persona in target_personas:
                p = self.other_personas[persona.name]
                current_conversation.append(
                    (
                        p.identity,
                        (
                            f"This month, I caught {agent_resource_num[p.agent_id]} tons"
                            " of fish!"
                        ),
                    ),
                )
                html_interactions.append(
                    "<strong>Framework</strong>:  This month, I caught"
                    f" {agent_resource_num[p.agent_id]} tons of fish!"
                )
        elif (
            self.cfg.inject_resource_observation
            and self.cfg.inject_resource_observation_strategy == "manager"
        ):
            # prepare report from pov of the manager
            current_report = {
                "date": current_time.strftime("%B %Y"),
                "individual_catches": {},
                "total": 0,
                "lake_fish_before": lake_fish_before,
                "lake_fish_after": lake_fish_after,
            }

            for persona in target_personas:
                p = self.other_personas[persona.name]
                fish_caught = agent_resource_num[p.agent_id]
                current_report["individual_catches"][p.identity.name] = fish_caught
                current_report["total"] += fish_caught

            self.fishing_reports.append(current_report)

            # Format all reports, with most recent first
            formatted_report = "Monthly Fishing Report:\n\n"

            for report in reversed(self.fishing_reports):
                formatted_report += f"=== {report['date']} ===\n"
                formatted_report += (
                    f"Fish in lake before harvest: {report['lake_fish_before']} tons\n"
                )
                for name, catch in report["individual_catches"].items():
                    formatted_report += f"- {name}: {catch} tons\n"
                formatted_report += f"Total catch: {report['total']} tons\n"
                formatted_report += (
                    f"Fish remaining in lake: {report['lake_fish_after']} tons\n\n"
                )

            current_conversation.append(
                (
                    PersonaIdentity("framework", "Mayor"),
                    (
                        f"Ladies and gentlemen, here is our fishing report history:\n\n{formatted_report}"
                    ),
                ),
            )

        max_conversation_steps = self.cfg.max_conversation_steps  # TODO

        current_persona = self.persona.identity

        while True:
            focal_points = [current_context]
            if len(current_conversation) > 0:
                # Last 4 utterances
                for _, utterance in current_conversation[-4:]:
                    focal_points.append(utterance)
            focal_points = self.other_personas[current_persona.name].retrieve.retrieve(
                focal_points, top_k=5
            )

            if current_persona.name.lower() == "jack":
                # Show the conversation history first
                print("\nConversation so far:")
                for persona, utterance in current_conversation:
                    print(f"{persona.name}: {utterance}")
                print("\nContext:", current_context)
                print("\nFocal points (relevant memories):")
                for point in focal_points:
                    print(f"- {point}")
                print("\n")

                # Get Jack's response
                utterance = input(f"Enter Jack's response: ")
                end_conversation = input("End conversation? (y/n): ").lower() == "y"
                next_name = (
                    input("Who should speak next?: ") if not end_conversation else None
                )
                h = f"<strong>Jack</strong>: {utterance}"
                html_interactions.append(h)
            else:
                utterance, end_conversation, next_name, h = (
                    prompt_converse_utterance_in_group(
                        self.model,
                        current_persona,
                        target_personas,
                        focal_points,
                        current_location,
                        current_time,
                        current_context,
                        self.conversation_render(current_conversation),
                    )
                )
                html_interactions.append(h)

            current_conversation.append((current_persona, utterance))

            if end_conversation or len(current_conversation) >= max_conversation_steps:
                # Log the complete conversation at the end
                self.model._log_and_print(
                    f"\n=== CONVERSATION SUMMARY: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ==="
                )
                self.model._log_and_print(f"Location: {current_location}\n")
                for persona, utterance in current_conversation:
                    self.model._log_and_print(f"{persona.name}: {utterance}")
                self.model._log_and_print("=== CONVERSATION END ===\n")
                break
            else:
                current_persona = self.other_personas[next_name].identity

        summary_conversation, h = prompt_summarize_conversation_in_one_sentence(
            self.model, self.conversation_render(current_conversation)
        )
        html_interactions.append(h)

        resource_limit, h = prompt_find_harvesting_limit_from_conversation(
            self.model, self.conversation_render(current_conversation)
        )
        html_interactions.append(h)

        for persona in target_personas:
            p = self.other_personas[persona.name]
            p.store.store_chat(
                summary_conversation,
                self.conversation_render(current_conversation),
                self.persona.current_time,
            )
            p.reflect.reflect_on_convesation(
                self.conversation_render(current_conversation)
            )

            if resource_limit is not None:
                p.store.store_thought(
                    (
                        "The community agreed on a maximum limit of"
                        f" {resource_limit} tons of fish per person."
                    ),
                    self.persona.current_time,
                    always_include=True,
                )

        print(f"Conversation length: {len(current_conversation)}")
        print(f"HTML interactions length: {len(html_interactions)}")

        return (
            current_conversation,
            summary_conversation,
            resource_limit,
            html_interactions,
        )

    def conversation_render(self, conversation: list[tuple[PersonaIdentity, str]]):
        return [(p.name, u) for p, u in conversation]
