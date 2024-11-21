from datetime import datetime

from simulation.persona.common import ChatObservation, PersonaOberservation


class HarvestingObs(PersonaOberservation):
    current_resource_num: int
    before_harvesting_resource_num: int
    agent_resource_num: dict[str, int]

    def __init__(
        self,
        phase: str,
        current_location: str,
        current_location_agents: dict[str, str],
        current_time: datetime,
        events: list,
        context: str,
        chat: ChatObservation,
        current_resource_num: int,
        agent_resource_num: dict[str, int],
        before_harvesting_sustainability_threshold: int,
        before_harvesting_resource_num: int,
    ) -> None:
        super().__init__(
            phase,
            current_location,
            current_location_agents,
            current_time,
            events,
            context,
            chat,
        )
        self.current_resource_num = current_resource_num
        self.before_harvesting_resource_num = before_harvesting_resource_num
        self.agent_resource_num = agent_resource_num
        self.before_harvesting_sustainability_threshold = (
            before_harvesting_sustainability_threshold
        )

    def __str__(self):
        """Should have ALL info nicely formatted"""
        output = []
        output.append(f"Phase: {self.phase}")
        output.append(f"Current Location: {self.current_location}")
        output.append(f"Current Time: {self.current_time}")
        output.append(f"Resources in Pool: {self.current_resource_num}")
        output.append(
            f"Resources Before Harvesting: {self.before_harvesting_resource_num}"
        )
        output.append(
            f"Sustainability Threshold: {self.before_harvesting_sustainability_threshold}"
        )

        if self.agent_resource_num:
            output.append("\nAgent Resources:")
            for agent, resources in self.agent_resource_num.items():
                output.append(f"  {agent}: {resources}")

        if self.current_location_agents:
            output.append("\nAgents at Locations:")
            for agent, location in self.current_location_agents.items():
                output.append(f"  {agent} at {location}")

        if self.events:
            output.append("\nEvents:")
            for event in self.events:
                output.append(f"  {event}")

        if self.chat:
            output.append("\nChat:")
            output.append(f"  {self.chat}")

        if self.context:
            output.append("\nContext:")
            output.append(f"  {self.context}")

        return "\n".join(output)
