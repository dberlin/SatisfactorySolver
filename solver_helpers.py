from modeler_models import ModelerNodeModel


def collect_vars(node_inputs, node_outputs, nodes):
    all_input_vars = []
    all_output_vars = []
    producer_output_vars = []
    # all_output_vars
    for (node_output_dict, node_input_dict, node) in zip(node_outputs.values(), node_inputs.values(), nodes):
        if node.Name == "AWESOME Sink":
            continue
        is_producer = len(node_input_dict) == 0 and len(node_output_dict) == 1
        is_consumer = len(node_output_dict) == 0 and len(node_input_dict) > 0
        all_input_vars.extend(node_input_dict.values())
        all_output_vars.extend(node_output_dict.values())
        if is_producer:
            producer_output_vars.extend(node_output_dict.values())
    return all_input_vars, all_output_vars, producer_output_vars


# These are the 1.0 resource limits, which we use to bound the producers of various sorts,
# even in the presence of no other limits.

class ResourceLimits:
    Iron = 92100
    Copper = 36900
    Limestone = 69900
    Coal = 42300
    Caterium = 15000
    CrudeOil = 12600
    Quartz = 13500
    Sulfur = 10800
    Bauxite = 12300
    Uranium = 2100
    Nitrogen = 12000
    SAM = 10200

    # Water is unlimited
    @classmethod
    def get_limit_for_node(cls, node: ModelerNodeModel):
        match node.Outputs[0].Part.Name:
            case "Coal":
                return cls.Coal
            case "Iron Ore":
                return cls.Iron
            case "Copper Ore":
                return cls.Copper
            case "Nitrogen Gas":
                return cls.Nitrogen
            case "Uranium Ore":
                return cls.Uranium
            case "Crude Oil":
                return cls.CrudeOil
            case "Sulfur":
                return cls.Sulfur
            case "Caterium Ore":
                return cls.Caterium
            case "Raw Quartz":
                return cls.Quartz
            case "Bauxite":
                return cls.Bauxite
            case "Limestone":
                return cls.Limestone
            case "SAM":
                return cls.SAM
            case "Uranium":
                return cls.Uranium
            case "Water":
                return 2 ** 32
        raise ValueError(f"Could not find resource limit for part {node.Outputs[0].Part.Name}")
