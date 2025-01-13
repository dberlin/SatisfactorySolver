# Copyright (c) 2024 Daniel Berlin and others
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from satisfactorysolver.modeler_models import ModelerNodeModel


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

    @staticmethod
    def get_resource_names():
        return [
            "Bauxite", "Caterium Ore", "Coal", "Copper Ore", "Crude Oil", "Iron Ore", "Limestone", "Nitrogen Gas",
            "Raw Quartz", "SAM", "Sulfur", "Uranium", "Water",
        ]

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
