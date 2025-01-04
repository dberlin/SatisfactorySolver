import itertools
from fractions import Fraction
from functools import cached_property
from typing import Optional, Self

from pydantic import BaseModel, model_validator, field_validator, computed_field, Field

from satisfactorysolver.helpers import validate_fraction_helper, rich_fraction_helper


class MachineModel(BaseModel):
    Name: str
    Tier: Optional[str] = None
    AveragePower: Optional[int] = None
    OverclockPowerExponent: Optional[Fraction] = None
    ProductionShardMultiplier: Optional[Fraction] = None
    ProductionShardPowerExponent: Optional[int] = None
    MaxProductionShards: Optional[int] = None
    ShowPpm: Optional[bool] = False
    DefaultMax: Optional[int] = None

    def __eq__(self, other):
        return self.Name == other.Name

    def __hash__(self):
        return hash(self.Name)

    @model_validator(mode='after')
    def index_name(self) -> Self:
        MachineByName.add(self.Name, self)
        return self

    @field_validator('OverclockPowerExponent', 'ProductionShardMultiplier', mode='before')
    @classmethod
    def validate_exponent(cls, exponent: str) -> Fraction | None:
        return validate_fraction_helper(exponent)


class MultiMachineMachineModel(BaseModel):
    Name: str
    PartsRatio: Optional[Fraction] = 1
    Default: Optional[bool] = False
    ShowPpm: Optional[bool] = False
    DefaultMax: Optional[int] = None

    def __eq__(self, other):
        return self.Name == other.Name

    def __hash__(self):
        return hash(self.Name)

    @field_validator('PartsRatio', mode='before')
    @classmethod
    def validate_exponent(cls, exponent: str) -> Fraction | None:
        return validate_fraction_helper(exponent)

    @field_validator('Name', mode='after')
    @classmethod
    def validate_name(cls, name: str) -> str:
        if MachineByName.get(name) is None:
            raise ValueError(f"Machine {name} does not exist")
        return name

    @computed_field
    @cached_property
    def machine(self) -> MachineModel | None:
        return MachineByName.get(self.Name)


class MultiMachineCapacityModel(BaseModel):
    Name: str
    PartsRatio: Optional[Fraction] = 1
    Default: Optional[bool] = False

    @field_validator('PartsRatio', mode='before')
    @classmethod
    def validate_exponent(cls, exponent: str) -> Fraction | None:
        return validate_fraction_helper(exponent)


class MultiMachineModel(BaseModel):
    Name: str
    ShowPpm: Optional[bool] = False
    DefaultMax: Optional[int] = None
    Machines: Optional[set[MultiMachineMachineModel]] = None
    Capacities: list[MultiMachineCapacityModel]

    def __eq__(self, other):
        return self.Name == other.Name

    def __hash__(self):
        return hash(self.Name)

    @model_validator(mode='after')
    def index_name(self) -> Self:
        MultiMachineByName.add(self.Name, self)
        return self


class PartModel(BaseModel):
    Name: str
    Tier: Optional[str] = None
    SinkPoints: Optional[int] = None

    def __eq__(self, other):
        return self.Name == other.Name

    def __hash__(self):
        return hash(self.Name)

    @model_validator(mode='after')
    def index_name(self) -> Self:
        PartByName.add(self.Name, self)
        return self


class RecipePartModel(BaseModel):
    Part: PartModel
    Amount: Fraction

    @field_validator('Part', mode='before')
    @classmethod
    def validate_part(cls, part_name: str) -> PartModel:
        part = PartByName.get(part_name)
        if not part:
            raise ValueError(f"Part {part_name} does not exist")
        return part

    @field_validator('Amount', mode='before')
    @classmethod
    def validate_amount(cls, amount: str) -> Fraction:
        return validate_fraction_helper(amount)

    def __rich_repr__(self):
        yield self.Part.Name
        yield rich_fraction_helper(self.Amount)


class RecipeModel(BaseModel):
    Name: str
    Parts: list[RecipePartModel]
    Machine: MachineModel | MultiMachineModel
    Tier: Optional[str] = None
    BatchTime: int | Fraction
    Alternate: Optional[bool] = False
    MinPower: Optional[int] = None

    def __eq__(self, other):
        return self.Name == other.Name

    def __hash__(self):
        return hash(self.Name)

    @field_validator('Machine', mode='before')
    @classmethod
    def validate_machine(cls, machine_name: str) -> MachineModel | MultiMachineModel:
        machine = MultiMachineByName.get(machine_name) or MachineByName.get(machine_name)
        if not machine:
            raise ValueError(f"Machine/MultiMachine {machine_name} does not exist")
        return machine

    @field_validator('BatchTime', mode='before')
    @classmethod
    def validate_batchtime(cls, amount: str) -> int | Fraction:
        return validate_fraction_helper(amount)

    @model_validator(mode='after')
    def index_name(self) -> Self:
        RecipeByName.add(self.Name, self)
        return self

    @computed_field
    @cached_property
    def Inputs(self) -> list[RecipePartModel]:
        return [part for part in self.Parts if part.Amount < 0]

    @computed_field
    @cached_property
    def Outputs(self) -> list[RecipePartModel]:
        return [part for part in self.Parts if part.Amount >= 0]

    def __rich_repr__(self):
        yield f"name", self.Name,
        yield f"inputs", self.Inputs,
        yield f"outputs", self.Outputs,
        yield f"machine", self.Machine.Name
        yield f"batchtime", rich_fraction_helper(self.BatchTime),


class AllDataModel(BaseModel):
    Machines: set[MachineModel]
    MultiMachines: set[MultiMachineModel]
    Parts: set[PartModel]
    Recipes: set[RecipeModel]


class ByNameRegistry[T]:
    def __init__(self):
        self._by_name = {}

    def add(self, name: str, model: T):
        self._by_name[name] = model

    def get(self, name: str) -> T:
        return self._by_name.get(name)


MachineByName = ByNameRegistry[MachineModel]()
MultiMachineByName = ByNameRegistry[MultiMachineModel]()
PartByName = ByNameRegistry[PartModel]()
RecipeByName = ByNameRegistry[RecipeModel]()


def validate_input_by_id_helper(data) -> dict[str, list[int]]:
    # Outputs have lists of empty lists sometimes
    # Awesome sink is a list of inputs instead of a dict
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        # We can't use a simple generator because we have to merge values
        result = {}
        for val in data[0]:
            if val[1] in result:
                result[val[1]].append(int(val[0]))
            else:
                result[val[1]] = [int(val[0])]
        return result
    return data


class ModelerNodeModel(BaseModel):
    Name: str
    ParentId: Optional[int] = Field(alias="Parent", default=None, repr=False)
    InputNodesById: Optional[dict[str, list[int]]] = Field(alias="Inputs", repr=False, default=None)
    Max: Optional[Fraction] = None
    Id: Optional[int] = None
    Machine: Optional[MachineModel] = None

    @staticmethod
    def _id_converter(item):
        return ModelerNodeById.get(item)

    @classmethod
    def _item_converter(cls, item) -> tuple[str, list[Self]]:
        return item[0], list(map(ModelerNodeModel._id_converter, item[1]))

    @field_validator("Max", mode="before")
    @classmethod
    def convert_fraction(cls, text) -> Fraction:
        return validate_fraction_helper(text)

    @field_validator("Machine", mode="before")
    @classmethod
    def convert_machine(cls, machine_name: str) -> MachineModel:
        return MachineByName.get(machine_name)

    @computed_field
    @cached_property
    def InputNodes(self) -> list[tuple[str, list[Self]]]:
        if self.InputNodesById:
            return list(map(self._item_converter, self.InputNodesById.items()))
        return []

    @computed_field
    @cached_property
    def Inputs(self) -> list[RecipePartModel]:
        if not self.Recipe:
            return []
        return self.Recipe.Inputs

    @computed_field
    @cached_property
    def Outputs(self) -> list[RecipePartModel]:
        if not self.Recipe:
            return []
        return self.Recipe.Outputs

    @computed_field
    @cached_property
    def Parent(self) -> Self:
        if self.ParentId:
            return self._id_converter(self.ParentId)
        return []

    @computed_field
    @cached_property
    def Recipe(self) -> RecipeModel:
        return RecipeByName.get(self.Name)

    @field_validator("InputNodesById", mode="before")
    @classmethod
    def validate_input_by_id(cls, data) -> dict[str, list[int]]:
        return validate_input_by_id_helper(data)

    @model_validator(mode="after")
    def index_by_id(self):
        ModelerNodeById.add(self)
        return self


class ModelerNodeById:
    _by_id = {}
    _next_id = itertools.count()

    @classmethod
    def add(cls, node: ModelerNodeModel):
        the_id = next(cls._next_id)
        cls._by_id[the_id] = node
        node.Id = the_id

    @classmethod
    def get(cls, node_id: int) -> ModelerNodeModel:
        return cls._by_id.get(node_id)


class ModelerFileModel(BaseModel):
    Version: str
    Outpost: Optional[int] = None
    Nodes: list[ModelerNodeModel] = Field(alias="Data")
