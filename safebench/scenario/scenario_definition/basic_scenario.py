from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


class BasicScenario(object):
    """
        Base class for user-defined scenario
    """
    def __init__(self, name, config, world):
        self.world = world
        self.name = name
        self.config = config

        self.ego_vehicles = None
        self.reference_actor = None
        self.other_actors = []
        self.other_actor_transform = []
        self.trigger_distance_threshold = None
        self.ego_max_driven_distance = 200

        
        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()
    

    def create_behavior(self, scenario_init_action):
        """
            This method defines the initial behavior of the scenario
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def update_behavior(self, scenario_action):
        """
            This method defines how to update the behavior of the actors in scenario in each step.
        """
        raise NotImplementedError(
                "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def initialize_actors(self):
        """
            This method defines how to initialize the actors in scenario.
        """
        raise NotImplementedError(
                "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def check_stop_condition(self):
        """
            This method defines the stop condition of the scenario.
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def clean_up(self):
        """
            Remove all actors
        """
        try:
            world = CarlaDataProvider.get_world()
            world_actor_ids = {actor.id for actor in world.get_actors()} if world is not None else set()
        except Exception:
            world_actor_ids = set()

        valid_actor_ids = []
        for actor in self.other_actors:
            try:
                if actor is None:
                    continue
                if world_actor_ids and actor.id not in world_actor_ids:
                    continue
                if not CarlaDataProvider.actor_id_exists(actor.id):
                    continue
                valid_actor_ids.append(actor.id)
            except Exception:
                continue

        if valid_actor_ids:
            try:
                CarlaDataProvider.remove_actors_by_ids(valid_actor_ids)
            except Exception:
                pass
        self.other_actors = []
